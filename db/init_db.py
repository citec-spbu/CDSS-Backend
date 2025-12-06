import logging
import os
import time

from db.postgres import DataManager

logger = logging.getLogger(__name__)


def initialize_database():
    print("Инициализация базы данных...")

    data_manager = DataManager()

    try:
        data_manager.initialization_db()
        print("База данных успешно инициализирована")

        if os.getenv("LOAD_MINZDRAV_DATA", "false").lower() == "true":
            from docs_processing.upload_files import sync_minzdrav_documents  # noqa: WPS433

            limit_env = os.getenv("LOAD_MINZDRAV_LIMIT")
            limit = int(limit_env) if limit_env else None
            force_reload = os.getenv("LOAD_MINZDRAV_FORCE").lower() == "true"
            push_embeddings = os.getenv("LOAD_MINZDRAV_PUSH_EMBEDDINGS").lower() == "true"

            print("Загрузка начальных данных Минздрава...")
            sync_minzdrav_documents(limit=limit, force_reload=force_reload, push_embeddings=push_embeddings)

    except Exception as e:
        logger.exception("Ошибка при инициализации базы данных: %s", e)
        # Повторная попытка через 5 секунд (не более 3 попыток)
        retry_left = int(os.getenv("INIT_DB_RETRY_COUNT"))
        if retry_left > 1:
            os.environ["INIT_DB_RETRY_COUNT"] = str(retry_left - 1)
            time.sleep(5)
            initialize_database()
        else:
            raise


if __name__ == "__main__":
    initialize_database()