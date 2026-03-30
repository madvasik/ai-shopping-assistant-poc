# Диаграмма потоков данных

```mermaid
flowchart LR
    subgraph browser["Браузер"]
        user["Пользователь"]
        session_store[("Сессия")]
    end

    subgraph server["Сервер"]
        api["API-слой"]
        sessions[("Активные сессии")]
        orch["Оркестратор"]
        llm_svc["LLM-сервисы"]
        retriever["Поиск по каталогу"]
        logger["Логирование"]
    end

    subgraph storage["Хранилище"]
        db[("Каталог товаров")]
        logfile[("Логи вызовов")]
    end

    openai["OpenAI API"]
    dashboard["Панель админа"]
    admin["Админ"]

    user -->|"сообщение"| api
    session_store -->|"идентификатор сессии"| api
    api -->|"история диалога"| sessions
    sessions -->|"история диалога"| api
    api -->|"сообщение + история"| orch
    orch --> llm_svc
    orch --> retriever
    llm_svc -->|"промпты"| openai
    openai -->|"ответы"| llm_svc
    retriever -->|"запрос"| db
    db -->|"товары"| retriever
    llm_svc --> logger
    logger --> logfile
    orch -->|"ответ"| api
    api -->|"ответ"| user
    logfile --> dashboard
    admin --> dashboard
```

## Что хранится

| Данные | Где | Как долго |
|--------|-----|-----------|
| Каталог товаров | SQLite-файл | Постоянно |
| Поисковый индекс | RAM | До перезапуска |
| Активные сессии + история диалога | RAM | До перезапуска, макс. 20 сообщений |
| Логи LLM-вызовов | JSON-файл | Последние 20 запросов |
| Идентификатор сессии | Браузер | До очистки браузера |

## Что не логируется

Персональные данные, API-ключи, IP-адреса пользователей, история диалогов между сессиями.
