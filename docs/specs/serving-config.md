# Spec: Serving / Config

## Запуск

`cd project && docker compose up --build`

| Сервис | Порт |
|--------|:----:|
| Widget API (FastAPI) | 8001 |
| Панель админа (Streamlit) | 8501 |

Внутри контейнера оба процесса управляются supervisor.

## Переменные окружения

| Переменная | Обязательная | Дефолт | Описание |
|-----------|:---:|--------|---------|
| `OPENAI_API_KEY` | **Да** | — | API-ключ OpenAI |
| `OPENAI_MODEL` | Нет | `gpt-4o-mini` | Идентификатор модели Chat Completions |
| `LLM_TEMPERATURE` | Нет | `0.4` | Температура для генерации |
| `LLM_TOP_P` | Нет | `0.95` | Top-p sampling |
| `PRODUCTS_DB_PATH` | Нет | `back/database/products.db` | Путь к SQLite |
| `PRODUCTS_TABLE` | Нет | `products` | Имя таблицы |
| `UPSTREAM_CHAT_URL` | Нет | `""` | URL внешнего chat API |

Файл: `project/.env` (скопировать из `project/.env.example`).

## Секреты

`OPENAI_API_KEY` — только через env, не коммитить в git, не логировать.

## Версии

| Компонент | Версия |
|-----------|--------|
| Python | 3.11+ |
| FastAPI | 0.115.6 |
| openai (Python SDK) | совместимо с используемым в `requirements.txt` |
| rank-bm25 | ≥ 0.2.2 |
| Streamlit | ≥ 1.37 |
| LLM модель | `gpt-4o-mini` по умолчанию (`OPENAI_MODEL`; для production зафиксировать snapshot-id при необходимости) |

## Контроль доступа по домену

Файл `widget/app/tenants.json`: список виджетов и разрешённых доменов для встраивания.
