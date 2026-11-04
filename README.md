# GitLab Handbook Agent

Мультиагентный RAG-сервис, который помогает сотруднику разобраться, как действовать
в рабочей ситуации, опираясь на публичный GitLab Handbook. Пользователь задаёт вопрос
обычным языком, система ищет инструкции, формирует ответ с источниками и проверяет
его перед выдачей. Диалог можно продолжить после перезапуска приложения.

Пример вопроса: «Я работаю в GitLab GmbH в Германии. Хочу взять обычный отпуск
на неделю. Что мне нужно сделать?»

Проект реализован как Python backend с HTTP API и CLI. Для ручного тестирования
доступен Swagger UI. Отдельный пользовательский фронтенд пока не реализован.

## Что реализовано

- Загрузка Markdown из Git-снимка публичного Handbook с сохранением метаданных,
  хешей и ссылок на исходники в конкретном коммите.
- Parent–child разбиение через LangChain: поиск по небольшим фрагментам и передача
  более крупных родительских фрагментов в контекст ответа.
- Гибридный dense + BM25 поиск в Qdrant, объединение результатов через RRF.
- Подготовка английского текста для BM25: очистка Markdown/HTML, нормализация,
  токенизация и стемминг. Исходный текст сохраняется для контекста и цитирования.
- Локальный cross-encoder reranker, который повторно ранжирует кандидатов поиска.
- LangGraph с Supervisor, Research Agent, Writer Agent и Guardrails.
- Раздельная настройка провайдера, модели и лимита ответа для каждой LLM-роли.
- Детерминированная проверка цитат и отдельная LLM-проверка смысла и применимости.
- Сохранение состояния в PostgreSQL через `AsyncPostgresSaver`, восстановление
  диалога по `thread_id` и продолжение прерванных запусков.
- FastAPI: диалоги, сообщения, `/chat`, debug, resume, liveness и readiness.
- Подготовка векторов в Colab/Kaggle и перенос в локальный Qdrant без повторного
  вычисления эмбеддингов; контрольные суммы и checkpoints для продолжения обработки.
- Автоматические тесты загрузки, чанкинга, поиска, графа, конфигурации моделей,
  переноса векторов, checkpoint и HTTP API, включая PostgreSQL-интеграцию.

## Архитектура

Подготовка базы выполняется отдельно от обработки пользовательских сообщений:

```mermaid
flowchart LR
    Git[GitLab Handbook: Git snapshot] --> Export[Markdown + metadata]
    Export --> Split[LangChain parent-child splitting]
    Split --> Dense[Dense embeddings]
    Split --> BM25[BM25 preprocessing]
    Dense --> Q[(Qdrant)]
    BM25 --> Q
    Split --> Parents[Parent texts + source links]
    Parents --> Q
```

Обработка сообщения:

```mermaid
flowchart TD
    Input[HTTP API / CLI] --> GuardIn[Input Guardrails]
    GuardIn --> Supervisor[Supervisor]
    Supervisor -->|research| Research[Research Agent]
    Supervisor -->|clarify / refuse| Save[Save conversation turn]
    Research --> Retrieval[Dense + BM25 / RRF / parent expansion / reranker]
    Retrieval --> Evidence{Sources found?}
    Evidence -->|yes| Writer[Writer Agent]
    Evidence -->|no| Save
    Writer --> GuardOut[Output Guardrails]
    GuardOut -->|one bounded rewrite| Writer
    GuardOut -->|checked response / clarify / refuse| Save
    Save --> DB[(PostgreSQL checkpoints)]
    Save --> Reply[Response with sources]
```

Схема показывает основной поток: ошибки моделей и поиска также завершаются
контролируемым ответом. После поиска Supervisor направляет выполнение к Writer
без дополнительного LLM-вызова. В успешном ходе без переписывания — четыре
LLM-вызова: маршрутизация, план поиска, написание и проверка ответа.

### Как устроен поиск

1. Research формирует 1–2 самостоятельных запроса с учётом сообщения и истории:
   `dense_query` и английский `sparse_query` для английского Handbook.
2. Qdrant получает до 100 child-кандидатов из каждой ветки dense/BM25 на запрос.
   RRF объединяет их в ранжированный список до 100 child-попаданий.
3. Из каждого запроса извлекаются до 30 уникальных parents. Результаты запросов
   чередуются и дедуплицируются по `parent_id`.
4. До 40 parents передаются reranker. Он сравнивает заголовок и текст parent с
   английской формулировкой запроса; при нескольких запросах берётся максимум оценок.
5. До 8 лучших parents передаются в граф. В API/CLI лимит контекста источников
   по умолчанию — 24 000 символов. Это настройки приложения, а не жёсткие константы.

Настройки чанкинга по умолчанию: parent — 3000 символов с overlap 300;
child — 600 символов с overlap 100. Векторизуются children, тексты parents тоже
хранятся в Qdrant и извлекаются по ID. Новые документы не индексируются при `/chat`.

### Модели

| Компонент | Модель по умолчанию | Где выполняется |
|---|---|---|
| Dense embeddings, 384 измерения | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Локально, FastEmbed/ONNX; GPU поддерживается при экспорте |
| Sparse retrieval | `Qdrant/bm25` | Подготовка на CPU, IDF и поиск в Qdrant |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` | Локально на CPU, FastEmbed/ONNX |
| Supervisor | `google/gemma-3-4b-it` | RouterAI API |
| Research | `qwen/qwen-2.5-7b-instruct` | RouterAI API |
| Writer | `openai/gpt-4o-mini` | RouterAI API |
| Output Guardrails: semantic review | `openai/gpt-4o-mini` | RouterAI API |

Четыре LLM-роли могут использовать один ключ RouterAI, но разные модели. Для каждой
роли также поддерживаются провайдеры `openrouter`, `openai` и `ollama`. RouterAI и
OpenRouter — разные сервисы: их ключи и настройки не взаимозаменяемы. Список моделей
в таблице — текущие defaults проекта, а не результат сравнительной оценки качества.

## Запуск

Нужны **Python 3.12**, Git, Docker Desktop с Docker Compose и ключ выбранного
LLM-провайдера. Проект проверялся на Python 3.12 в Windows/PowerShell.
Команды ниже выполняются из корня репозитория.

### 1. Зависимости и конфигурация

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

В `.env` укажите `ROUTERAI_API_KEY` или настройте другого провайдера для ролей.
Ключи не нужны для загрузки Handbook, локальной векторизации и индексации.
Файл `.env.example` содержит шаблон без API-ключей.

### 2. Qdrant и PostgreSQL

```powershell
docker compose up -d --wait
```

Qdrant доступен на `127.0.0.1:6333`, PostgreSQL — на `127.0.0.1:5433`.
Данные сохраняются в Docker volumes `qdrant_storage` и `postgres_checkpoints`
(Docker Compose добавляет к именам префикс проекта). Для API таблицы создаются
автоматически при старте. Векторную коллекцию нужно подготовить отдельно.

### 3. Подготовка индекса

Для нового клона репозитория:

```powershell
.\.venv\Scripts\python.exe ingest_handbook.py
.\.venv\Scripts\python.exe chunk_handbook.py
.\.venv\Scripts\python.exe hybrid_search.py index --batch-size 8 --sparse-batch-size 128
```

Эта команда считает dense на CPU и создаёт полный индекс. Для большого корпуса
подготовку dense можно перенести на GPU в Colab/Kaggle: см. [руководство](docs/RUNBOOK.md#векторизация-в-colabkaggle).

Если готовый экспорт уже есть, вместо вычисления эмбеддингов импортируйте его:

```powershell
.\.venv\Scripts\python.exe vector_transfer.py import --input data/handbook_vectors_bm25
```

Дождитесь завершения с `status: ready` и `alias: handbook`. Готовый индекс в volume
используется повторно при следующих запусках. Файлы данных и векторов не входят
в Git-репозиторий: новый пользователь создаёт индекс или импортирует свой экспорт.

### 4. HTTP API

```powershell
.\.venv\Scripts\python.exe run_api.py
```

Откройте [Swagger UI](http://127.0.0.1:8000/docs). Сервер должен оставаться запущенным.
`run_api.py` настраивает совместимый с psycopg `SelectorEventLoop` на Windows и
запускает один worker. API привязан к `127.0.0.1`; авторизация не реализована.
Граф и клиенты переиспользуются, локальные модели загружаются лениво при первом поиске.

### CLI

```powershell
.\.venv\Scripts\python.exe run_agent.py --thread-id vacation-demo "Как оформить отпуск в GitLab GmbH в Германии?"
.\.venv\Scripts\python.exe run_agent.py --thread-id vacation-demo "А как отметить его в системе?"
```

Без вопроса включается интерактивный режим; для выхода — `/exit`:

```powershell
.\.venv\Scripts\python.exe run_agent.py --thread-id vacation-demo
```

CLI и API работают с одними checkpoint и индексом. Не отправляйте сообщения
одновременно в один диалог через оба интерфейса.

## HTTP API

| Метод и путь | Назначение |
|---|---|
| `POST /threads` | Создать диалог с UUID или заданным `thread_id` |
| `GET /threads` | Список диалогов, включая существующие CLI-диалоги |
| `GET /threads/{thread_id}/messages` | Полная история завершённых ходов |
| `POST /chat` | Новое сообщение, проверенный ответ и ссылки на источники |
| `GET /threads/{thread_id}/debug` | Сохранённые состояния, черновики и диагностика |
| `POST /threads/{thread_id}/resume` | Продолжить прерванный запуск |
| `GET /healthz` | Проверить, что HTTP-процесс отвечает |
| `GET /readyz` | Проверить PostgreSQL, Qdrant и готовность индекса |

В Swagger сначала выполните `POST /threads`:

```json
{"thread_id": "demo-vacation", "title": "Оформление отпуска"}
```

Затем `POST /chat`:

```json
{
  "thread_id": "demo-vacation",
  "message": "Я работаю в GitLab GmbH в Германии. Как оформить обычный отпуск на неделю?"
}
```

Продолжение диалога — ещё один `/chat` с тем же ID и новым `message`.
Ответ появляется в **Server response → Response body**. Поля ответа:
`thread_id`, `checkpoint_id`, `status`, `answer`, `sources`, `reason`, `trace`.

| `status` | Значение |
|---|---|
| `ok` | Ответ прошёл проверки и содержит ссылки на источники |
| `clarify` | Нужен существенный факт о ситуации пользователя |
| `refuse` | Вопрос вне области, недостаточно подтверждений или произошла ошибка |

`reason` различает содержательные отказы (`no_evidence`, `out_of_scope`,
`grounding_failed`) и ошибки инфраструктуры (`llm_error`, `retrieval_error`).
Подробности HTTP-кодов, пагинации, resume и конфигурации — в [RUNBOOK](docs/RUNBOOK.md).

## История и диагностика

`AsyncPostgresSaver` сохраняет состояние после шагов графа. В контекст моделей
передаются последние 6 сообщений, суммарно до 12 000 символов, и новый вопрос
до 6000 символов. На время уточнений сохраняется исходный `pending_question`;
это не суммаризация всей переписки. Полная история завершённых ходов доступна
через API и не ограничена шестью сообщениями.

Чтобы уменьшить повторные уточнения, промпты запрещают переспрашивать уже
сообщённые обстоятельства. После ответа на уточнение повторный `clarify` от
Supervisor код направляет в Research; решение `refuse` этим правилом не меняется.

```powershell
.\.venv\Scripts\python.exe run_agent.py --thread-id vacation-demo --state-history --output data/debug/vacation-demo.json
```

В state видны `queries`, `sources`, `draft`, `feedback`, `supervisor_decision`,
`guardrail_review` и `retrieval_diagnostics`. Для поиска сохраняются исходные и
итоговые ранги, RRF/reranker scores и этап сбоя. Просмотр не вызывает LLM.

## Тестирование

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

По умолчанию интеграционные тесты PostgreSQL пропускаются. Для их запуска:

```powershell
$env:TEST_CHECKPOINT_DATABASE_URL = 'postgresql://handbook:handbook_local@localhost:5433/handbook?connect_timeout=10'
.\.venv\Scripts\python.exe -m unittest discover -v
```

Тесты используют настоящий LangGraph, embedded Qdrant и, при включении интеграции,
PostgreSQL. Проверяются маршруты, цитаты, отказ при недостатке данных, восстановление,
перенос векторов, HTTP-контракты, полная история и ограничения конкурентных запросов.
LLM-вызовы в автоматических тестах подменены; это проверка поведения кода, а не
оценка качества ответов реальных моделей.

## Данные и границы проекта

На снимке commit `6b25f87e3ff771138322e193606d9697bce7f06c` подготовлены
**4118 документов, 18 583 parents и 114 491 children**. Эти числа относятся к
конкретному снимку; при повторной загрузке актуального Handbook они могут измениться.

Источники: [публичный Handbook](https://handbook.gitlab.com/handbook/) и его
[Git-репозиторий](https://gitlab.com/gitlab-com/content-sites/handbook).
Закрытый Internal Handbook, внешние Google Docs, PDF и изображения не загружаются.
Hugo-включения не разворачиваются. В корпусе могут быть архивные и устаревшие страницы.

Проверка дословной цитаты не доказывает, что правило подходит пользователю.
Отдельный verifier проверяет смысл, но тоже может ошибаться. Пока нет размеченного
eval-набора и измеренных retrieval/answer quality metrics. Reranking не гарантирует
выбор правильной страны; длинные parents могут обрезаться токенизатором reranker.
Промпты снижают риск prompt injection, но не являются гарантией защиты.

API рассчитан на локальный запуск: один активный запрос к агенту, без пользовательской
авторизации. Из-за локальных ONNX-моделей на машинах с 8 ГБ RAM возможна нехватка
памяти. Настройки малых batch и отключения reranker описаны в руководстве.

## Файлы проекта

| Файлы | Назначение |
|---|---|
| `ingest_handbook.py`, `chunk_handbook.py` | Загрузка корпуса и parent–child разбиение |
| `hybrid_search.py`, `bm25_text.py` | Векторизация, BM25, индексация и RRF |
| `handbook_retriever.py`, `reranking.py` | Retrieval для графа и reranking |
| `handbook_graph.py`, `node_models.py` | Узлы, промпты, проверки и LLM-провайдеры |
| `checkpoint_store.py`, `inspect_graph.py` | PostgreSQL checkpoint и просмотр state |
| `api.py`, `api_store.py`, `run_api.py` | FastAPI, каталог диалогов и запуск сервера |
| `run_agent.py` | CLI с историей, debug и resume |
| `vector_transfer.py`, `handbook_embeddings_gpu.ipynb` | Экспорт/импорт и GPU notebook |
| `test_*.py` | Автоматические тесты |
| `compose.yaml`, `.env.example`, `requirements*.txt` | Инфраструктура и зависимости |

## Дальнейшее развитие

- Разметить eval-набор рабочих ситуаций и измерять retrieval recall, groundedness
  и корректность уточнений, особенно для правил разных стран.
- Добавить пользовательский чат с источниками и панелью разработчика.
- Показывать этапы обработки и собирать обратную связь на ответы.
- Записать демонстрацию реального сценария с продолжением диалога и проверкой источников.

Подробные команды: [RUNBOOK](docs/RUNBOOK.md).
Состав репозитория для публикации: [PUBLISHING](docs/PUBLISHING.md).
