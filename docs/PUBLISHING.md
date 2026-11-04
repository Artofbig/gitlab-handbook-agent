# Какие файлы публиковать на GitHub

Для текущего проекта публикуйте исходники, тесты, notebook без результатов выполнения,
шаблон настроек и документацию. Большие выгрузки для запуска создаются по инструкции.

## Точный состав

```text
.gitignore
.env.example
README.md
compose.yaml
requirements.txt
requirements-gpu.txt
handbook_embeddings_gpu.ipynb
docs/
    RUNBOOK.md
    PUBLISHING.md

api.py
api_store.py
bm25_text.py
checkpoint_store.py
chunk_handbook.py
handbook_graph.py
handbook_retriever.py
hybrid_search.py
ingest_handbook.py
inspect_graph.py
node_models.py
reranking.py
run_agent.py
run_api.py
vector_transfer.py

test_api.py
test_bm25.py
test_checkpoints.py
test_chunking.py
test_graph.py
test_hybrid.py
test_ingest.py
test_node_models.py
test_reranking.py
test_transfer.py
```

Все корневые `.py` из текущего проекта включены. Нужен и `inspect_graph.py`,
и `api_store.py`: приложение импортирует их. Тесты импортируют вспомогательные
объекты друг из друга, поэтому публикуйте весь набор `test_*.py`.

## Что оставить локально

| Файл/каталог | Причина |
|---|---|
| `.env` и другие локальные `.env.*` | Реальные ключи и настройки; публикуется только `.env.example` |
| `.venv/`, `venv/` | Установленные зависимости, воспроизводятся из requirements |
| `.cache/` | Git-кеш, модели и временные скрипты диагностики |
| `data/` целиком | Корпус, chunks, векторы, ZIP-архивы и сохранённые пользовательские диалоги |
| `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/` | Автоматически созданные служебные файлы |
| Веса `.onnx`/`.safetensors`, `.vhdx`, логи | Локальные артефакты и диски контейнеров |
| `.git/` при ручной загрузке файлов | Служебная история локального Git; её не загружают как обычную папку |

Docker volumes Qdrant/PostgreSQL и `docker_data.vhdx` в репозиторий не нужны.
Публичный репозиторий содержит способ воспроизвести индекс, а не вашу рабочую базу.
При переносе ноутбука после выполнения очистите outputs и проверьте его metadata.
Текущий notebook подготовлен без сохранённых результатов выполнения.

## Добавление через Git

В текущем `.gitignore` перечислены локальные данные, окружение и артефакты. Для
добавления именно текущего набора исходников из PowerShell:

```powershell
git add -- .gitignore .env.example README.md compose.yaml requirements.txt requirements-gpu.txt handbook_embeddings_gpu.ipynb docs
Get-ChildItem -File -Filter '*.py' | ForEach-Object { git add -- $_.Name }
git diff --cached --stat
git status --short
```

Эти команды не создают commit и не отправляют файлы в GitHub. Перед commit проверьте
список: в нём не должно быть `.env`, `.cache/`, `data/` и `.venv/`.
Для проверки правил исключения:

```powershell
git check-ignore .env .cache/models/example.onnx data/handbook_vectors_bm25/manifest.json
```

`.gitignore` действует при работе через Git. При ручной загрузке через браузер
выбирайте файлы по списку выше: браузерный upload сам по себе не применяет локальный
`.gitignore`. Если файл уже был закоммичен раньше, добавление правила не удаляет
его из истории. В этом проекте список отслеживаемых файлов можно проверить командой
`git ls-files`.

## Презентация проекта

README описывает реализованные возможности и отдельно перечисляет планы. После
записи демо добавьте проверенное видео или ссылку в README. Для статичных материалов
можно создать `docs/media/` и положить туда скриншоты Swagger, ответа с источниками
и диаграммы. Не используйте для этого исходные debug-выгрузки с полной историей.

Демо и оценка качества пока не заявлены как готовые. Не добавляйте вымышленные
метрики, бейдж CI без настроенного workflow или утверждение о production readiness.
