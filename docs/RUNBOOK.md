# Запуск и эксплуатация

Основная инструкция — в [README](../README.md). Ниже дополнительные варианты
подготовки данных, настройки API и диагностика.

## Окружение

В README команды приведены для Windows/PowerShell. На Linux/macOS после
`python -m venv .venv` используйте `.venv/bin/python` вместо
`.\.venv\Scripts\python.exe`. Значения окружения задаются по правилам вашей оболочки.
Для Windows используйте `run_api.py`: он выбирает SelectorEventLoop, необходимый
асинхронному psycopg. Docker Desktop может выполнять контейнеры через WSL 2.

`.env` автоматически читают CLI агента и HTTP API. Утилиты `hybrid_search.py` и
`vector_transfer.py` читают переменные процесса; для другого Qdrant передайте
`--url` явно или задайте `QDRANT_URL`/`QDRANT_API_KEY` в окружении.

## Повторное использование корпуса

Загрузчик работает с отдельным bare Git-кешем `.cache/handbook.git`. Для обновления
выполните `ingest_handbook.py`, затем `chunk_handbook.py` и индексацию. Для экспорта
уже скачанного Git-снимка без сети:

```powershell
.\.venv\Scripts\python.exe ingest_handbook.py --offline
```

Указатели `data/handbook/CURRENT.json` и `data/parent_child/CURRENT.json` ссылаются
на успешно завершённые snapshots. При ошибке предыдущий опубликованный снимок остаётся.
Повторная индексация создаёт новую физическую коллекцию Qdrant и переключает alias
только после загрузки и проверки количества записей. Старые коллекции автоматически
не удаляются и занимают место. Это полная пересборка снимка, а не incremental update.

Небольшой технический smoke test индекса:

```powershell
.\.venv\Scripts\python.exe hybrid_search.py index --limit 64
.\.venv\Scripts\python.exe run_agent.py --alias handbook_sample "О чём GitLab Handbook?"
```

`--limit` создаёт alias `handbook_sample`. Это проверка механики на первых chunks;
ответы на произвольные рабочие вопросы требуют полного корпуса. Для API такого
индекса задайте `QDRANT_ALIAS=handbook_sample` в `.env`. После полного импорта
используйте `QDRANT_ALIAS=handbook`.

## Векторизация в Colab/Kaggle

1. Локально загрузите документы и выполните `chunk_handbook.py`.
2. Подготовьте входной архив:

   ```powershell
   .\.venv\Scripts\python.exe vector_transfer.py pack
   ```

   Он создаст `data/handbook_bm25_input.zip`: скрипты, GPU-зависимости и
   `snapshot/parents.jsonl`, `snapshot/children.jsonl`. Секреты и кеш моделей не включаются.

3. Откройте [handbook_embeddings_gpu.ipynb](../handbook_embeddings_gpu.ipynb),
   включите GPU и интернет, загрузите архив в Colab Files или Kaggle Dataset.
   Выполните ячейки установки и экспорта. Notebook устанавливает отдельный
   `requirements-gpu.txt`; не устанавливайте GPU-зависимости поверх локального CPU-окружения.
4. Dense вычисляется на GPU, BM25 — на CPU. `LIMIT=None` означает весь корпус.
   Для пробного запуска задайте `LIMIT=64` и другую папку вывода.
5. Скачайте `handbook_vectors_bm25.zip` и распакуйте его в
   `data/handbook_vectors_bm25/`. Внутри должны быть `manifest.json`, `parents.jsonl`,
   `children.jsonl` и `vectors-*.jsonl.gz`. Отдельные `.gz` распаковывать не нужно.
6. Запустите локальный Qdrant и импорт:

   ```powershell
   docker compose up -d --wait
   .\.venv\Scripts\python.exe vector_transfer.py import --input data/handbook_vectors_bm25
   ```

Архив содержит документы и вычисленные векторы, а не готовую файловую базу Qdrant.
Импорт создаёт коллекцию и индексы, проверяет хеши, размеры, ID и parent–child связи.
После успешного импорта папка экспорта не нужна для выполнения запросов.
При этом локальная dense-модель всё равно нужна для векторизации новых вопросов.

В notebook checkpoint экспорта записывается по завершённым gzip shards. Повторный
запуск с тем же OUTPUT использует `resume`; batch можно уменьшить, но источник и
конфигурация должны совпадать. Временный диск notebook может исчезнуть при сбросе
сессии: для продолжения нужен сохранённый каталог экспорта, а не только код notebook.

Для совместимого старого экспорта dense можно переиспользовать через `REUSE_DENSE`
в notebook или `--reuse-dense` у `vector_transfer.py export`. Нужен существующий
`manifest.json`; модель dense, размерность, данные и порядок ID должны совпадать.
Поддержка старых SPLADE-экспортов сохранена для переноса dense, текущий default — BM25.

Для продолжения прерванного локального импорта используйте имя физической коллекции,
которое напечатано в логе `Collection: ...`:

```powershell
.\.venv\Scripts\python.exe vector_transfer.py import --input data/handbook_vectors_bm25 --resume handbook_REPLACE_WITH_ACTUAL_ID
```

Замените `handbook_REPLACE_WITH_ACTUAL_ID` на фактическое имя. `--resume` импорта
принимает имя коллекции; `--resume` экспорта является флагом.

## Подготовка BM25

`bm25_text.py` преобразует Markdown/HTML в текст, оставляя заголовки, подписи ссылок,
содержимое таблиц и code blocks. Убирает HTML-комментарии, script/style, URL-адреса,
Hugo shortcodes; нормализует Unicode NFKC и пробелы. Далее FastEmbed выполняет
токенизацию, lowercasing и английский Snowball stemming. Лемматизации нет.

Стоп-слова не удаляются, чтобы сохранять в том числе `not`, `no`, `must`.
Средняя длина рассчитывается по реальным проиндексированным children после этой
обработки; параметры сохраняются в конфигурации индекса. Qdrant применяет IDF
через `Modifier.IDF`. Очистка применяется только к BM25, не к исходным цитатам
и не к тексту dense embeddings. BM25 сам по себе не понимает смысл отрицаний.

## Настройки API

Значения ниже — defaults API, которые можно переопределить в `.env`:

| Переменная | Значение |
|---|---|
| `API_MODEL_THREADS` | `1` |
| `API_RERANK_ENABLED` | `true` |
| `API_RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` |
| `API_RERANK_BATCH_SIZE` | `1` |
| `API_TOP_K` | `8` |
| `API_CANDIDATES` | `100` |
| `API_RETRIEVAL_K` | `30` |
| `API_RERANK_LIMIT` | `40` |
| `API_MAX_CONTEXT_CHARS` | `24000` |

После изменения конфигурации перезапустите сервер. Допускается один активный
`/chat`/`resume` на процесс; второй получает `429` и `Retry-After: 1`. Между экземплярами
API используется PostgreSQL advisory lock по thread ID. Это не распределённая
очередь заданий. Для другого порта: `python run_api.py --port 8001`.

`/readyz` проверяет доступность БД, наличие готового индекса и конфигурацию моделей.
Он не делает платных LLM-вызовов, не проверяет валидность ключа у провайдера и не
гарантирует, что ONNX-модели загрузятся при текущем объёме свободной RAM.

По умолчанию PostgreSQL: `postgresql://handbook:handbook_local@localhost:5433/handbook?connect_timeout=10`.
Для другой БД задайте `CHECKPOINT_DATABASE_URL`. При настройке нового локального
контейнера согласуйте `POSTGRES_PASSWORD` в Compose и пароль в URL. Переменная
окружения не меняет пароль уже созданного пользователя в существующем volume.

## HTTP-коды и история

- `201`: создан диалог; `200`: проверенный ответ, уточнение или содержательный отказ.
- `404`: ID диалога не найден. Сначала создайте его через `POST /threads`.
- `409`: ID уже существует, диалог занят другим API-процессом или требует resume.
- `422`: неверный JSON/параметры; сообщение должно содержать 1–6000 символов.
- `429`: агент уже обрабатывает другой запрос.
- `502`: граф вернул `llm_error`; `503`: `retrieval_error`, сбой backend или неготовый `/readyz`.

Обработанная ошибка графа возвращает поля `status`, `answer`, `reason`, `sources`,
`trace`. Ошибка backend может вернуть `detail.code`. Причина `refuse` определяется
по `reason`, а не только по HTTP-коду.

`GET /threads` поддерживает `limit` (до 100) и `offset`, сортировка по последней
активности. Каталог хранится в `handbook_api_threads`. Существующие CLI-диалоги
подхватываются из checkpoint-таблиц.

`GET /threads/{id}/messages` возвращает завершённые ходы от старых к новым.
`limit` (по умолчанию 50, максимум 100) и `offset` считаются **в ходах**:
один ход содержит сообщение пользователя и ответ ассистента. Время сообщения —
время завершения хода. Черновики не попадают в эту историю.

`GET /threads/{id}/debug` возвращает до 100 последних checkpoints в хронологическом
порядке. Параметры: `history=false` для последнего состояния, `limit` до 500 и
`before=<checkpoint_id>`. `next_before` указывает курсор следующей старой страницы.
Для пустого нового диалога — пустой список или `latest: null`.

`POST /chat` не является идемпотентным. После сетевого обрыва сначала проверьте
`messages` и `debug`, чтобы не отправить завершённое сообщение повторно. Если запуск
прерван, вызовите `POST /threads/{id}/resume`. Для завершённого графа resume возвращает
последний ответ без нового хода. Незавершённый внешний LLM-вызов может повториться.

CLI-эквиваленты:

```powershell
.\.venv\Scripts\python.exe run_agent.py --thread-id demo --state
.\.venv\Scripts\python.exe run_agent.py --thread-id demo --state-history --output data/debug/demo.json
.\.venv\Scripts\python.exe run_agent.py --thread-id demo --resume
```

Checkpoints содержат состояния и структурированные результаты узлов. Сырые
HTTP-ответы провайдера и внутренние рассуждения модели не сохраняются.
Для необязательной трассировки LangSmith поддерживаются переменные из `.env.example`;
при включении трассировка может отправлять вопросы и документы во внешний сервис.

## Частые ошибки

| Симптом | Что проверить |
|---|---|
| Swagger: `Failed to fetch` | Запущен ли `run_api.py`, доступен ли `/healthz`; вкладка Swagger может остаться открытой после остановки сервера |
| PowerShell не находит `.venv` или скрипт | Выполняйте команды из корня репозитория, а не из `data/` |
| `LengthFinishReasonError` | Провайдер остановил ответ по длине; проверьте `*_MAX_TOKENS` соответствующей роли и контекстное окно модели |
| `retrieval_error` | В debug проверьте `retrieval_diagnostics.stage`: index, embedding_load, hybrid_search или rerank |
| ONNX `Fail` / allocation error | Свободную RAM, batch, число потоков и доступность файлов модели; один тип `Fail` не доказывает нехватку памяти |
| `grounding_failed` | Сравните `draft`, `feedback`, `guardrail_review` и `diagnostics.citation_issues` |
| Экспорт не продолжается | Тот ли OUTPUT и сохранён ли `manifest.json`, совпадают ли данные/модели |

`*_MAX_TOKENS` отправляется провайдеру как жёсткий максимум генерации, включая JSON.
Промпт о краткости не заменяет этот лимит. В шаблоне заданы 300/600/2500/1000 токенов
для Supervisor/Research/Writer/Guardrails; для более длинных результатов можно,
например, увеличить их до 1024/1024/4000/2048, учитывая ограничения выбранных моделей.

Для ограниченной RAM в API уменьшите `API_RERANK_BATCH_SIZE` до 1 и оставьте
`API_MODEL_THREADS=1`. Чтобы отключить reranker, задайте `API_RERANK_ENABLED=false`.
В CLI доступны `--rerank-batch-size 1 --threads 1` и `--no-rerank`.
Это снижает нагрузку, но не гарантирует размещение моделей в любой свободной памяти.

## Где хранятся данные

| Хранилище | Содержимое |
|---|---|
| `.cache/` | Git-кеш и скачанные ONNX-модели |
| `data/handbook`, `data/parent_child` | Полные документы и snapshots чанков |
| `data/handbook_vectors_bm25` | Переносимый экспорт документов и вычисленных векторов |
| Docker volume `qdrant_storage` | Коллекции, векторы, payload и индексы Qdrant |
| Docker volume `postgres_checkpoints` | Состояния графа и каталог диалогов |

Физическое расположение Docker volumes на диске хоста зависит от настроек Docker
Desktop. Внутри контейнера Qdrant данные находятся в `/qdrant/storage`.
Экспорт из Colab и рабочая база — отдельные копии. В Git публикуются код и инструкции,
локальные данные и виртуальные диски Docker в репозиторий не входят.
