from __future__ import annotations

import dataclasses
import datetime as dt
import io
import logging
import os
import time
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pdfplumber
import pyexcel as pe
import requests

from db.postgres import DataManager

logger = logging.getLogger(__name__)

API_URL = "https://apicr.minzdrav.gov.ru/api.ashx"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json;charset=UTF-8",
    "Origin": "https://apicr.minzdrav.gov.ru",
    "Referer": "https://apicr.minzdrav.gov.ru/",
}

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"

if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception as e:
        print(str(e))

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE"))

CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP"))
MIN_CHUNK_LENGTH = int(os.getenv("PDF_MIN_CHUNK_LENGTH"))

REQUEST_TIMEOUT = int(os.getenv("MINZDRAV_REQUEST_TIMEOUT"))
MAX_RETRIES = int(os.getenv("MINZDRAV_MAX_RETRIES"))
RETRY_DELAY = int(os.getenv("MINZDRAV_RETRY_DELAY"))


@dataclasses.dataclass
class ClinicalDocument:
    raw_id: str
    base_id: str
    version: int
    title: str
    mcb: Optional[str]
    age_category: Optional[str]
    developer: Optional[str]
    publish_date: Optional[dt.date]
    source_url: str
    specialties: Optional[str] = None

    @property
    def storage_id(self) -> str:
        return self.raw_id


class MinzdravClient:
    FILTER_PAYLOAD = {"filter": {"status": [1], "search": "", "year": "", "specialties": []}}

    def fetch_documents(self) -> List[ClinicalDocument]:
        logger.info("Загрузка списка клинических рекомендаций...")
        response = self._request(
            method="POST",
            params={"op": "GetJsonClinrecsFilterV2Excel"},
            json=MinzdravClient.FILTER_PAYLOAD,
        )
        sheets = pe.get_array(file_content=response.content, file_type="xlsx")
        logger.info("Получено строк в реестре: %s", len(sheets))

        docs_by_base: Dict[str, List[ClinicalDocument]] = defaultdict(list)

        for row in sheets:
            doc = self._parse_row(row)
            if not doc:
                continue
            docs_by_base[doc.base_id].append(doc)

        selected: List[ClinicalDocument] = []
        for base_id, versions in docs_by_base.items():
            latest = max(versions, key=lambda d: (d.publish_date or dt.date.min, d.version))
            selected.append(latest)

        selected.sort(key=lambda d: d.base_id)
        logger.info("Отобрано актуальных документов: %s", len(selected))
        return selected

    def download_pdf(self, doc: ClinicalDocument) -> bytes:
        logger.debug("Скачивание PDF для документа %s", doc.base_id)
        response = self._request(
            method="GET",
            params={"op": "GetClinrecPdf", "id": doc.base_id},
        )
        return response.content

    def _request(self, method: str, params: Dict, json: Optional[Dict] = None) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.request(
                    method=method,
                    url=API_URL,
                    params=params,
                    headers=DEFAULT_HEADERS,
                    json=json,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                return response
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Ошибка обращения к API (попытка %s/%s): %s",
                    attempt,
                    MAX_RETRIES,
                    exc,
                )
                time.sleep(RETRY_DELAY)
        raise RuntimeError(f"Не удалось выполнить запрос к API Минздрава: {last_exc}") from last_exc

    @staticmethod
    def _parse_row(row: Sequence) -> Optional[ClinicalDocument]:
        if not row or len(row) < 7:
            return None

        raw_id = str(row[0]).strip()
        if not raw_id or raw_id == "ID":
            return None

        has_pdf = str(row[5]).strip().lower() != "нет"
        if not has_pdf:
            return None

        title = str(row[1]).strip() if row[1] else ""
        mcb = str(row[2]).strip() if row[2] else None
        age_category = str(row[3]).strip() if row[3] else None
        developer = str(row[4]).strip() if row[4] else None
        publish_date = row[6] if isinstance(row[6], dt.datetime) else None
        specialties = str(row[7]).strip() if len(row) > 7 and row[7] else None

        base_id, version = _parse_base_id(raw_id)
        source_url = f"https://cr.minzdrav.gov.ru/clin-rec/{base_id}"

        return ClinicalDocument(
            raw_id=raw_id,
            base_id=base_id,
            version=version,
            title=title,
            mcb=mcb,
            age_category=age_category,
            developer=developer,
            publish_date=publish_date.date() if publish_date else None,
            source_url=source_url,
            specialties=specialties,
        )


def _parse_base_id(raw_id: str) -> Tuple[str, int]:
    if "_" not in raw_id:
        return raw_id, 1
    base, suffix = raw_id.split("_", maxsplit=1)
    try:
        version = int(suffix)
    except ValueError:
        version = 1
    return base, version


def _normalize_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    filtered = [ln for ln in lines if ln]
    normalized = " ".join(filtered)
    return " ".join(normalized.split())


def _chunk_text(text: str) -> Iterator[str]:
    if not text:
        return
    chunk_size = max(CHUNK_SIZE, MIN_CHUNK_LENGTH)
    overlap = min(CHUNK_OVERLAP, chunk_size // 2)
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LENGTH:
            yield chunk
        if end == text_len:
            break
        start = max(end - overlap, start + 1)


def _extract_chunks(pdf_data: bytes) -> List[Dict]:
    chunks: List[Dict] = []
    with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("Не удалось извлечь текст со страницы %s: %s", page_number, exc)
                continue

            normalized = _normalize_text(raw_text)
            if len(normalized) < MIN_CHUNK_LENGTH:
                continue

            for chunk_index, chunk_text in enumerate(_chunk_text(normalized)):
                chunks.append(
                    {
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                    }
                )
    return chunks


def _prepare_metadata(doc: ClinicalDocument, chunk: Dict) -> Dict:
    page = chunk["page"]
    chunk_index = chunk["chunk_index"]
    chunk_id = f"{doc.base_id}-p{page}-c{chunk_index}"
    recommendation_number = f"{doc.base_id}-p{page}"

    metadata = {
        "document_id": doc.base_id,
        "raw_document_id": doc.raw_id,
        "document_name": doc.title,
        "version": doc.version,
        "section": f"Страница {page}",
        "page": page,
        "chunk_index": chunk_index,
        "chunk_id": chunk_id,
        "recommendation_number": recommendation_number,
        "source_url": doc.source_url,
        "mcb": doc.mcb,
        "age_category": doc.age_category,
        "developer": doc.developer,
        "publish_date": doc.publish_date.isoformat() if doc.publish_date else None,
        "specialties": doc.specialties,
    }
    return metadata


def _batched(items: List[Dict], batch_size: int) -> Iterator[List[Dict]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _push_embeddings(chunks: List[Dict], doc: ClinicalDocument) -> None:
    if not chunks:
        logger.info("Документ %s пропущен: текст не найден", doc.base_id)
        return

    logger.info("Отправка %s чанков документа %s в embedding-service", len(chunks), doc.base_id)
    for batch in _batched(chunks, EMBEDDING_BATCH_SIZE):
        payload = {
            "texts": [item["text"] for item in batch],
            "metadata": [_prepare_metadata(doc, item) for item in batch],
            "task": "retrieval.passage",
            "dimensions": EMBEDDING_DIMENSIONS,
        }
        response = requests.post(EMBEDDING_SERVICE_URL, json=payload, timeout=REQUEST_TIMEOUT)
        try:
            response.raise_for_status()
        except Exception as exc:
            logger.error("Ошибка при сохранении эмбеддингов документа %s: %s", doc.base_id, exc)
            raise
        logger.debug("embedding-service ответ: %s", response.json())


def sync_minzdrav_documents(
    *,
    limit: Optional[int] = None,
    force_reload: bool = False,
    push_embeddings: bool = True,
) -> None:
    logging.basicConfig(level=logging.INFO)
    client = MinzdravClient()
    data_manager = DataManager()

    existing_ids = set()
    if not force_reload:
        existing_ids = data_manager.get_existing_document_ids()
        logger.info("В БД найдено %s документов", len(existing_ids))

    documents = client.fetch_documents()
    if limit:
        documents = documents[:limit]
        logger.info("Ограничено к обработке %s документов", limit)

    total = len(documents)
    for index, doc in enumerate(documents, start=1):
        logger.info("Документ %s/%s: %s", index, total, doc.title)

        if not force_reload and doc.storage_id in existing_ids:
            logger.info("Документ %s уже есть в БД, пропуск", doc.raw_id)
            continue

        try:
            pdf_bytes = client.download_pdf(doc)
            data_manager.save_document(
                doc_id=doc.storage_id,
                title=doc.title,
                mcb=doc.mcb or "NULL",
                age_category=doc.age_category or "Взрослые",
                developer=doc.developer or "NULL",
                placement_date=doc.publish_date or dt.date.today(),
                data=pdf_bytes,
            )
            logger.info("Документ %s сохранён", doc.raw_id)

            if push_embeddings:
                chunks = _extract_chunks(pdf_bytes)
                _push_embeddings(chunks, doc)
        except Exception as exc:
            logger.exception("Ошибка обработки документа %s: %s", doc.raw_id, exc)
            continue


if __name__ == "__main__":
    sync_minzdrav_documents()

