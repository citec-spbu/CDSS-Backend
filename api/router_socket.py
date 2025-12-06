import fastapi
import json
import os
import logging
from typing import Any, Dict, List, Optional

import asyncpg
import httpx
from fastapi import WebSocket, WebSocketDisconnect
from services.chat_service import ChatSessionManager

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"

logger = logging.getLogger()

if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception as e:
        print(str(e))


socket_router = fastapi.APIRouter()
session_manager = ChatSessionManager()

logger = logging.getLogger("example")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL")
RERANK_SERVICE_URL = os.getenv("RERANK_SERVICE_URL")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")

VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST")
VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT"))
VECTOR_DB_NAME = os.getenv("VECTOR_DB_NAME")
VECTOR_DB_USER = os.getenv("VECTOR_DB_USER")
VECTOR_DB_PASSWORD = os.getenv("VECTOR_DB_PASSWORD")

RETRIEVAL_LIMIT = int(os.getenv("RAG_RETRIEVAL_LIMIT"))


@socket_router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    session_id = None

    try:
        session_id = session_manager.create_session()
        await websocket.send_text(json.dumps({"type": "session_created", "session_id": session_id}, ensure_ascii=False))

        history = session_manager.get_history(session_id)
        await websocket.send_text(json.dumps({"type": "history", "messages": history}, ensure_ascii=False))

        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "chat_message":
                    user_query = msg.get("query", "").strip()
                    if not user_query:
                        continue

                    session_manager.add_message(session_id, "user", user_query)

                    # работа с моделью
                    response = await get_response(user_query)

                    session_manager.add_message(session_id, "bot", response)

                    await websocket.send_text(
                        json.dumps({"type": "chat_message", "role": "bot", "content": response}, ensure_ascii=False))

                if msg.get("type") == "history":
                    history = session_manager.get_history(session_id)
                    await websocket.send_text(json.dumps({"type": "history", "messages": history}, ensure_ascii=False))
            except Exception as e:
                await websocket.send_text(json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False))
    except WebSocketDisconnect:
        session_manager.remove_session(session_id)


async def _fetch_embedding(user_query: str) -> Optional[List[float]]:
    embed_request = {
        "texts": [user_query],
        "task": "retrieval.query",
        "dimensions": 1024
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(EMBEDDING_SERVICE_URL, json=embed_request, timeout=30.0)
            response.raise_for_status()
            embedding_data = response.json()
            embedding = embedding_data["embedding"][0]
            logger.info("Эмбеддинг получен. Размер: %s", len(embedding))
            return embedding
    except (httpx.HTTPError, KeyError, IndexError) as exc:
        logger.error("Ошибка при генерации эмбеддинга: %s", exc)
        return None


async def _fetch_similar_texts(embedding: List[float]) -> List[Dict[str, Any]]:
    if not embedding:
        return []

    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
    query = """
        SELECT id,
               content,
               metadata->>'document_name'  AS document_name,
               metadata->>'source_url'     AS source_url,
               metadata->>'recommendation_number' AS recommendation_number,
               1 - (embedding <=> $1::vector) AS similarity
        FROM chunks
        ORDER BY similarity DESC
        LIMIT $2;
    """

    conn: Optional[asyncpg.Connection] = None
    try:
        conn = await asyncpg.connect(
            host=VECTOR_DB_HOST,
            port=VECTOR_DB_PORT,
            database=VECTOR_DB_NAME,
            user=VECTOR_DB_USER,
            password=VECTOR_DB_PASSWORD,
        )
        records = await conn.fetch(query, embedding_str, RETRIEVAL_LIMIT)
        logger.info("Найдено %s похожих фрагментов", len(records))
        return [
            {
                "id": record["id"],
                "text": record["content"],
                "document_name": record["document_name"],
                "source_url": record["source_url"],
                "recommendation_number": record["recommendation_number"],
                "similarity": record["similarity"],
            }
            for record in records
        ]
    except asyncpg.PostgresError as exc:
        logger.error("Ошибка при работе с векторной БД: %s", exc)
        return []
    finally:
        if conn:
            await conn.close()


async def _rerank_results(user_query: str, passages: List[str]) -> Optional[Dict[str, Any]]:
    if not passages:
        return None

    rerank_request = {
        "query": user_query,
        "passages": passages,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(RERANK_SERVICE_URL, json=rerank_request, timeout=30.0)
            response.raise_for_status()
            rerank_data = response.json()
            logger.info("Результаты rerank получены")
            return rerank_data
    except httpx.HTTPError as exc:
        logger.error("Ошибка при rerank: %s", exc)
        return None


def _format_response(rerank_data: Dict[str, Any], passages: List[Dict[str, Any]]) -> str:
    results = rerank_data.get("reranked_results") if rerank_data else None
    if not results:
        return "Не удалось найти релевантные рекомендации по вашему запросу."

    lines: List[str] = []
    for idx, item in enumerate(results, start=1):
        text = item.get("text", "").strip()
        score = item.get("score")
        matched = next((p for p in passages if p["text"] == text), None)
        source_parts = []
        if matched:
            if matched.get("document_name"):
                source_parts.append(matched["document_name"])
            if matched.get("recommendation_number"):
                source_parts.append(f"рекомендация {matched['recommendation_number']}")
        source = ", ".join(source_parts) if source_parts else "Источник не указан"
        link = matched.get("source_url") if matched else None

        line = f"{idx}. {text}"
        if score is not None:
            line += f"\n   Оценка релевантности: {score:.2f}"
        line += f"\n   {source}"
        if link:
            line += f"\n   Ссылка: {link}"
        lines.append(line)

    return "\n\n".join(lines)


def _build_prompt(user_query: str, reranked_results: List[Dict[str, Any]]) -> str:
    context_lines: List[str] = []
    for idx, item in enumerate(reranked_results, start=1):
        doc_name = item.get("document_name") or "Неизвестный документ"
        rec_number = item.get("recommendation_number") or "—"
        source_url = item.get("source_url") or "Источник не указан"
        score = item.get("score")
        text = item.get("text", "").strip()

        score_display = f"{score:.2f}" if score is not None else "—"

        context_lines.append(
            f"{idx}. Документ: {doc_name} (рекомендация {rec_number}) | score={score_display}\n"
            f"   Ссылка: {source_url}\n"
            f"   Текст: {text}"
        )

    context_block = "\n\n".join(context_lines)
    instructions = (
        "Ты — медицинский ассистент. Ответь на вопрос врача строго на основе приведённых фрагментов. "
        "Если данных недостаточно, так и скажи. В ответе обязательно укажи использованные источники "
        "(название документа, номер рекомендации и ссылку). Не выдумывай информации."
    )

    prompt = (
        f"{instructions}\n\n"
        f"Вопрос врача: {user_query}\n\n"
        f"Релевантные фрагменты:\n{context_block}\n\n"
        "Сформируй структурированный ответ (краткий вывод + список рекомендаций с дозировками/мероприятиями, если они есть)."
    )
    return prompt


async def _call_llm(prompt: str, user_query: str, reranked_results: List[Dict[str, Any]]) -> Optional[str]:
    if not LLM_SERVICE_URL:
        logger.warning("LLM_SERVICE_URL не задан. Ответ будет сформирован без обращения к LLM.")
        return None

    payload = {
        "prompt": prompt,
        "query": user_query,
        "context": [
            {
                "text": item.get("text"),
                "document_name": item.get("document_name"),
                "recommendation_number": item.get("recommendation_number"),
                "source_url": item.get("source_url"),
                "score": item.get("score"),
            }
            for item in reranked_results
        ],
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(LLM_SERVICE_URL, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            llm_answer = data.get("answer") or data.get("response") or data.get("result")
            logger.info("Ответ от LLM получен")
            return llm_answer
    except httpx.HTTPError as exc:
        logger.error("Ошибка при обращении к LLM сервису: %s", exc)
        return None


def _merge_answer_with_sources(llm_answer: Optional[str], reranked_results: List[Dict[str, Any]]) -> str:
    sources_lines: List[str] = []
    for idx, item in enumerate(reranked_results, start=1):
        doc = item.get("document_name") or "Неизвестный документ"
        rec = item.get("recommendation_number") or "—"
        link = item.get("source_url") or "Ссылка не указана"
        score = item.get("score")
        score_display = f"{score:.2f}" if score is not None else "—"
        sources_lines.append(f"{idx}. {doc}, рекомендация {rec} (score={score_display})\n   {link}")

    sources_block = "\n".join(sources_lines)

    if llm_answer:
        return f"{llm_answer}\n\nИсточники:\n{sources_block}"

    # fallback ответ
    fallback_intro = "Не удалось получить итоговый ответ от LLM. Вот релевантные фрагменты:"
    fallback_lines: List[str] = []
    for idx, item in enumerate(reranked_results, start=1):
        doc = item.get("document_name") or "Неизвестный документ"
        rec = item.get("recommendation_number") or "—"
        link = item.get("source_url") or "Ссылка не указана"
        text = item.get("text", "").strip()
        fallback_lines.append(
            f"{idx}. {text}\n   Источник: {doc}, рекомендация {rec}\n   {link}"
        )
    fallback_block = "\n\n".join(fallback_lines)
    return f"{fallback_intro}\n\n{fallback_block}"


async def get_response(user_query: str) -> str:
    # получение эмбеддинга
    embedding = await _fetch_embedding(user_query)
    if embedding is None:
        return "Не удалось получить эмбеддинг для запроса. Повторите попытку позже."

    # db retrieval
    passages = await _fetch_similar_texts(embedding)
    if not passages:
        return "Релевантные рекомендации не найдены в базе данных."

    rerank_data = await _rerank_results(user_query, [p["text"] for p in passages])
    if rerank_data is None:
        return "Не удалось выполнить переранжирование результатов. Попробуйте повторить запрос позже."

    reranked_results_raw = rerank_data.get("reranked_results") or []
    matched_results: List[Dict[str, Any]] = []
    for item in reranked_results_raw:
        text = item.get("text", "")
        matched = next((p for p in passages if p["text"] == text), None)
        if matched:
            enriched = matched.copy()
            enriched["score"] = item.get("score")
            enriched["rank"] = item.get("rank")
            matched_results.append(enriched)

    if not matched_results:
        return _format_response(rerank_data, passages)

    prompt = _build_prompt(user_query, matched_results)
    llm_answer = await _call_llm(prompt, user_query, matched_results)

    return _merge_answer_with_sources(llm_answer, matched_results)