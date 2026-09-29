# MinerU Precision — отдельный opt-in пайплайн документов

**Не интегрирован с `/api/recognize` / OCR этикеток.** Эта ветка отправляет PDF/сканы в облако mineru.net **только после запуска `work --allow-upload` оператором**; ключ в `MINERU_TOKEN`, не в репозитории/логе/артефактах. Проверьте законность обработки и хранение данных у провайдера (в частности signed OSS URL и юрисдикцию). Не используйте для частных документов без разрешения. На реальном аккаунте сервис и качество не проверялись.

[Официальная документация Precision API](https://mineru.net/apiManage/docs) (сверено: 2026-09-29): `/api/v4/file-urls/batch` → PUT в подписанный OSS URL (24 ч) → callback или `GET /api/v4/extract-results/batch/{batch_id}` → zip по `full_zip_url`. `files[].is_ocr=true`, внешний `language=cyrillic|east_slavic` и `model_version=pipeline|vlm`. Документация перечисляет оба языковых пакета для pipeline/vlm; практическая точность VLM на RU не проверена. Flash/Agent API тут **не используется**, равно как URL-режим `/extract/task`. В очереди один файл на batch (ниже лимита 50 в одном запросе), сплит PDF по 200 страницам **и** 200 000 000 байт. Исходный PDF ограничен 1 ГБ/5000 страниц; изображения передаются без DPI-рекодирования — DPI-нормализация ещё не реализована.

## Установка и запуск (Python 3.12, без локального MinerU)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -c constraints.txt -r requirements-mineru.txt
mkdir -m 700 /private/path/mineru-jobs       # каталог БД, spool и results, НЕ web root
.venv/bin/python -m wineid.mineru_jobs --work-dir /private/path/mineru-jobs enqueue public-book.pdf scan.png
# После проверки прав/приватности и затрат, явное разрешение egress:
export MINERU_TOKEN='<secret>'
.venv/bin/python -m wineid.mineru_jobs --work-dir /private/path/mineru-jobs work --allow-upload --loop
.venv/bin/python -m wineid.mineru_jobs --work-dir /private/path/mineru-jobs status
```

CLI `work` без `--loop` делает **один шаг**, не ждёт завершения всего PDF; для фонового воркера запускайте `--loop` под supervisor. Рекомендуется отдельный single-worker процесс, свой лимит CPU/RAM/disk, запрещённый public доступ к рабочему каталогу. Нет egress из основного FastAPI. `--model vlm`, `--language east_slavic`, `--no-ocr`, `--formula`, `--no-table` — явные абляции, новый профиль создаёт новый cache key; для сканов по умолчанию `pipeline/cyrillic/is_ocr=true`. Никакого автоматического fallback между моделями: для сравнения задать два профиля и измерить на независимом GT.

Каждый chunk получает `data_id=SHA-256(bytes)`; локальный cache key включает также профиль OCR. **MinerU не гарантирует дедупликацию POST по `data_id`.** Поэтому после неопределённого результата POST задача помечается `needs_review` и **не отправляется повторно автоматически**. Состояние `queue_full` (`-60009`) повторяется с backoff, `daily_quota` (`-60018`) требует решения оператора/следующего дня, `extract_failed` (`-60010`) помечает failed. Подписанный upload URL повторно используется только в пределах 23 часов; после истечения — `needs_review`. У `GET` и PUT ограниченные retries, у POST нет слепых retry. Один процесс обрабатывает за раз один шаг; лимит активных локальных задач 100. Для production за reverse proxy нужны firewall/allowlist egress, supervisor, мониторинг состояний, уборка архивов и резервное копирование приватной БД. У провайдера высокая приоритетная квота 1000 страниц/сутки, не безусловный hard-limit: согласуйте бюджет до включения воркера.

## Callback (предпочтительный режим)

```bash
export MINERU_UID='<uid from mineru.net>' MINERU_CALLBACK_SEED='<random 32+ ASCII characters>'
# Этот URL должен быть доступен провайдеру по HTTPS, TLS завершать на своём reverse proxy:
.venv/bin/python -m wineid.mineru_jobs --work-dir /private/path/mineru-jobs enqueue public-book.pdf \
  --callback-url https://ocr.example.org/callbacks/mineru
.venv/bin/python -m wineid.mineru_jobs --work-dir /private/path/mineru-jobs work --allow-upload --loop \
  --callback-url https://ocr.example.org/callbacks/mineru
.venv/bin/python -m wineid.mineru_jobs --work-dir /private/path/mineru-jobs callback --host 127.0.0.1 --port 8081
```

`POST /callbacks/mineru` принимает только ≤70KB JSON `{content,checksum}`; проверяет `SHA256(uid+seed+**исходная строка** content)` постоянновременным сравнением и совпадение `batch_id`+`data_id` с БД. ACK возвращается **после** durable update, без скачивания zip в HTTP-потоке; повторная доставка идемпотентна. Не публиковать порт 8081 напрямую; наружу только TLS + сетевой allowlist/WAF с размерными ограничениями. Если callback утрачен, воркер делает контрольный poll только для устаревших задач (после 30 мин). Смена seed во время активных задач потребует ручного согласования. В callback URL не передавать секреты. Callback серверу нужен только `MINERU_UID`/`MINERU_CALLBACK_SEED`, а не `MINERU_TOKEN`.

## Артефакты и качество

`results/<key>.zip` — оригинальный архив (храните приватно). В `results/<key>/`: `full.md`, `content_list.json` из `*_content_list.json`, опциональные `layout.json`, `model.json`, `normalized.txt` (NFKC, lowercase, ё→е, пробелы), `qa.json` (хеши, флаги пустого текста/структуры). Только allowlisted HTTPS hosts для подписанных upload/download URL; редиректы отклоняются. ZIP проверяется на zip-slip, symlink, дубли, размеры и количество записей; **не** используется `extractall`. Извлечённый markdown и JSON — недоверенные данные; RAG должен экранировать/проверять типы блоков и ссылки. `qa.empty_text` — не confidence и не CER.

Для CER/WER по ручной буквальной разметке страниц (не по 200-страничному документу целиком): JSONL `key,gt_path,gt_sha256,font_class,split,gt_source`, `gt_path` внутри каталога manifest. Выполнить `.venv/bin/python -m wineid.mineru_eval gt.jsonl --results-dir /private/path/mineru-jobs/results --split validation`. Сравнивать печатный/художественный текст и заголовки отдельно, включать пропавшие результаты в знаменатель. Порог авто-fallback не выбирался; не отправляйте данные другому провайдеру без отдельного согласия. Без независимой разметки отчёт не доказывает качество.

**Не закрыто:** live API canary с разрешёнными файлами и токеном, DPI/preprocess изображений, автоматическое объединение chunks с page-offset в единый документ, мониторинг/алерты, cleanup/retention и нагрузочные/квотные испытания. Основной wine RC [план финализации](plans/FINALIZATION.md) этим не блокируется и не изменяется.
