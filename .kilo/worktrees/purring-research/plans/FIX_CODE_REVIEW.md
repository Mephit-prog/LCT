# План закрытия код-ревью (CODE_REVIEW.md, 2026-09-25)

Цель: исправить все 6 P1 и 12 P2 замечаний из `plans/CODE_REVIEW.md`, синхронизировать
документацию с кодом и подтвердить зелёные тесты (`pytest`, `compileall`, `pyflakes`).

## P1 — критичные

### P1-1 · `wineid/train_clip.py` — веса негативов «тот же производитель ×2» в torch-пути
- В `clip_loss(logits_t, temperature)` добавить параметр `producers: list[str]`.
- Реализовать симметричный взвешенный InfoNCE, идентичный numpy-эталону
  `info_nce_loss()`: матрица весов `weights[i,j]=2.0` для негативов одного производителя,
  1.0 иначе; `scaled = logits/temperature * weights`, диагональ возвращается к исходной;
  loss = 0.5 * (CE по строкам + CE по столбцам).
- В `train()` передавать уже вычисленный `producers` (строка 271) в `clip_loss`.
- Обновить докстринг модуля (строка 10) — теперь соответствует реализации.

### P1-2 · `wineid/catalog_export.py` + `wineid/train_clip.py` — контракт пути изображений
- `images.csv`: колонка `path` и так относительна корня `uploads` (голый basename от
  `catalog_media.image_path`), но `train()` резолвит относительно директории `images.csv`.
- Добавить параметр `images_root: Path | None = None` в `train()`/`evaluate_recall()`/
  `load_image_bytes()`; дефолт `images_csv.parent` (обратная совместимость с тестами).
- CLI `wineid/train_clip.py`: флаг `--images-root` (required при `--out`, отличном от `uploads`).
- Докстринги обоих модулей: зафиксировать «`path` в `images.csv` относителен корня
  `uploads`, переданного в `catalog_export --uploads`; обучению указывать `--images-root`».
- `IMPLEMENTATION.md` §train_clip: описать новый параметр.

### P1-3 · `wineid/train_clip.py` — загрузчик/инференс-путь обученного адаптера
- Новый формат чекпоинта: `torch.save({'meta': {lora_blocks, rank, alpha, dropout,
  revision}, 'state': best_state}, out)`.
- Новая функция `load_adapter(model, checkpoint_path, device=None) -> dict`:
  загружает meta, вызывает `attach_lora(...)` с сохранёнными гиперпараметрами,
  применяет `state` через `model.load_state_dict(..., strict=False)`; возвращает meta
  (включая `temperature` из state).
- Документировать путь инференса в докстринге модуля и в `IMPLEMENTATION.md`.
- Тест: stub-модель (nn.Module с `vision_model.encoder.layers[i].self_attn.{q,k,v,out}_proj`
  как nn.Linear) — round-trip attach → save → load; `skipUnless(find_spec('torch'))`.

### P1-4 · `wineid/server.py` — аутентификация и ограничение потоков
- `WINE_TOKEN` env: если задан, POST `/v1/recognize` и `/v1/eval/predict` требуют
  `Authorization: Bearer <token>`; иначе `401 {"status":"error","slug":None,
  "reason_codes":["unauthorized"]}`. `/ready` остаётся открытым.
- Ограничение параллельных запросов: `threading.BoundedSemaphore(WINE_MAX_WORKERS,
  default 8)` в Handler; в начале `do_POST` неблокирующий acquire → иначе
  `503 {"status":"error","reason_codes":["busy"]}`; release в `finally`.
- `create_handler(pipeline, timeout=8.5)` — параметр таймаута для тестов slow-body.
- `main()`: чтение `WINE_TOKEN`, `WINE_MAX_WORKERS`.

### P1-5 · `wineid/train_clip.py` — привязка LoRA к структуре HF CLIP
- Новая `inspect_vision_modules(model) -> dict`: обходит `vision_model.encoder.layers`,
  собирает реальные имена атрибутов self_attn/layer (JSON-отчёт для smoke-проверки).
- `attach_lora(..., projection_map=None)`: по умолчанию HF-CLIP-маппинг
  (`('q_proj','k_proj','v_proj','out_proj')` в `self_attn`), при отсутствии — попытка
  найти проекции прямо на layer; иначе `RuntimeError` с перечнем найденных атрибутов.
- CLI: флаг `--verify-only` (инспекция архитектуры и выход без обучения).
- Докстринг + `IMPLEMENTATION.md`: «перед обучением обязательна smoke-проверка маппинга
  на jina-clip-v2».

### P1-6 · `wineid/search.py` + `requirements.txt` — единый разрешённый fuzzy-бэкенд
**Решение пользователя: зависимости НЕ меняем — `fuzzywuzzy` остаётся.**
- `fuzzywuzzy` (GPL-2.0) фиксируется как единственный production-бэкенд;
  GPL-риск документируется в докстринге `search.py` и комментарием в
  `requirements.txt`.
- Fallback `difflib` помечается как **offline/non-production** («NOT
  FuzzyWuzzy-equivalent») и при активации выдаёт `warnings.warn` — ранжирование
  больше не меняется тихо. `Index.backend` по-прежнему отражает активный бэкенд.

## P2 — улучшения

### P2-7 · `wineid/canonical.py` — `canonical_query()` без выдуманного продюсера
- При `producer is None` строка строится без хвоста «, производитель …» (шаблон без
  producer), а не с плейсхолдером «Неизвестный». Не менять `canonical_text()`
  (контракт «name_ru и producer обязательны» сохраняется для индекса/тренировки).
- Тест в `tests/test_canonical.py`.

### P2-8 · `wineid/search.py` — удалить мёртвое поле `Candidate.fuzzy_rank`
- Убрать поле из dataclass и конструктора (потребителей нет — проверено поиском).
- Проверить, что сериализация `asdict(candidate)` и тесты не зависят от ключа.

### P2-9 · `wineid/ocr.py` — задокументировать незаполняемые `boxes`/`confidence`
- Оставить поля (совместимость схемы результата, тест `test_baseline` проверяет
  `confidence is None`), добавить комментарий в dataclass: «резерв контракта, не
  заполняется».

### P2-10 · `requirements.txt` — удалить `python-Levenshtein` (не используется)
- Удалить строку; проверить, что `search.levenshtein` — собственная реализация.

### P2-11 · неиспользуемые импорты
- `wineid/catalog_export.py:19` — удалить `from PIL import Image`.
- `wineid/clip_eval.py:14` — удалить `import numpy as np` (если реально не используется
  ниже — перепроверить pyflakes после правки).
- `wineid/train_clip.py:19` — удалить `import hashlib`; локальные
  `import json # noqa: F401` / `import time # noqa: F401` (строки 200–201) — удалить.
- `tests/test_search_v2.py:3` — убрать `load_catalog` из импорта.

### P2-12 · `wineid/canonical.py` — `VOLUME_RE` для объёмов 4+ знаков
- Расширить: `(?<!\d)(?:0[,.]\d{1,2}|(?!19\d\d|20\d\d)\d{3,4})(?!\d)\s*(?:л|мл|litre|литр)?`
  (3–4 цифры, кроме 19xx/20xx — те остаются vintage; «1000 мл» распознаётся, «2024»
  не поглощается).
- Тесты: `name_tokens('Вино 1000 мл') == ('вино',)`, `name_tokens('Рислинг 2021')`
  по-прежнему `('рислинг',)`, `extract_vintage('2024')` без изменений.

### P2-13 · негативные тесты HTTP/OCR/pipeline — новый `tests/test_http_negative.py`
- HTTP: 413 при `Content-Length > MAX_BODY`; отсутствующий/нечисловой `Content-Length`;
  chunked body (без Content-Length); 404 на неизвестном пути; `/ready` → 503 при
  пустом индексе; slow-body (через `create_handler(timeout=...)`); 401 при отсутствии
  токена / 200 при корректном Bearer (при `WINE_TOKEN`).
- OCR: `upstream_5xx`/`upstream_4xx`/`rate_limited` → `ocr_provider_error`;
  `invalid_response`; `no_text` → `unreadable`; `timeout/deadline`.
- Pipeline: `ocr_not_configured` (без OCR, ручной bbox); deadline между шагами
  (SlowOCR, исчерпывающий секунды).

### P2-14 · атомарная публикация без hardlink
- `wineid/visual.py:128` и `wineid/clip_zero_shot.py:158`: обернуть `os.link` в
  try/except OSError → fallback `os.rename(temporary, out)` (после повторной проверки
  `out.exists()`); комментарий про ФС без hardlink.

### P2-15 · лимит байт до чтения файла целиком
- `wineid/data.py::audit.inspect()` — проверять `path.stat().st_size > MAX_BYTES` до
  `read_bytes()` (ошибка `oversized image`).
- `wineid/catalog_export.py::image_filters()` — аналогичный пре-чек
  (`MAX_BYTES` импортируется из `.roi`) → `oversized`.
- `wineid/train_clip.py::load_image_bytes()` — пре-чек `stat().st_size > MAX_BYTES`
  до чтения.

### P2-16 · `wineid/server.py` — catch-all 500 без деталей
- После существующего except: `except Exception: reply(500, {'status':'error',
  'slug':None, 'reason_codes':['internal']})` (без деталей в ответе; при необходимости
  краткий лог в stderr без содержимого тела).

### P2-17 · `wineid/media.py` — комментарий про casefold-ключ групп
- Комментарий у строки 53: намеренное схлопывание регистра в ключе
  `derivative_groups` — «гипотеза суффикса», регистрозависимые различия не разделяются.

### P2-18 · `wineid/search.py` — пред-фильтр для fuzzy-расширения пула
- В цикле `extra` (строка 194) применять дешёвый токенный пред-фильтр: кандидат
  участвует, если хотя бы один токен документа разделяет префикс (первые 3 символа)
  с токеном запроса. Комментарий: сложность снижается с O(N·snippets·docs) до
  O(|пул-кандидатов|·snippets·docs); для каталога большего размера — отдельная задача.
- Тест: типичная опечатка («Алиготэ» vs «Алиготе») всё ещё попадает в пул.

## Синхронизация документации
- `IMPLEMENTATION.md` обновлён: §6.3 (веса производителей реализованы в torch-пути),
  контракт пути `--images-root`, `load_adapter()` для инференса, требование
  smoke-проверки LoRA-маппинга на jina-clip-v2 (`--verify-only`), политика
  fuzzy-бэкенда (fuzzywuzzy GPL-2.0 остаётся, difflib-fallback не тихий).

## Статус закрытия (2026-09-26)

Все пункты ревью (6×P1 + 12×P2) закрыты. Отклонение от исходного плана — по
решению пользователя: **P1-6** — зависимости не менялись; `fuzzywuzzy` (GPL-2.0)
остаётся единственным production-бэкендом, риск задокументирован в `search.py`
и `requirements.txt`, difflib-fallback помечен «offline only» и выдаёт
`warnings.warn` при активации (ранжирование больше не меняется тихо).

Итоговая верификация:
- `python3 -m pytest --timeout=30 -q` → **82 passed** (было 65; +17 новых тестов,
  включая `tests/test_http_negative.py` — 12 негативных HTTP/OCR/pipeline кейсов).
- `python3 -m compileall -q wineid tests` → OK.
- `pyflakes wineid tests` → чисто (exit 0).
- Предупреждение «fuzzywuzzy is not installed» — ожидаемое поведение нового
  fallback-уведомления в среде без fuzzywuzzy.

Осталось на будущее (не код): smoke-проверка LoRA-маппинга на реальной jina-clip-v2
(`python3 -m wineid.train_clip --verify-only --allow-remote-code`), калибровка
блендера/порогов на независимом validation.