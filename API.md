# API интеграции (FastAPI backend)

**Статус: экспериментальный исследовательский прототип.** Это тонкий HTTP-слой
поверх существующих `wineid.pipeline.Pipeline` и `wineid.search.Index`. По умолчанию основной API **отказывается** от автоматического принятия,
а eval возвращает best-effort top-1 для валидных изображений. Доступна явная
опциональная интеграция CLIP/бленда; `score`/cosine — не вероятности,
независимого GT/метрик качества нет. Backend не «делает вид», что умеет распознавать.

## Запуск

```bash
python3.12 -m pip install -c constraints.txt -r requirements-api.txt
python3.12 -m wineid.api           # uvicorn, по умолчанию 127.0.0.1:8080
```

OpenAPI/Swagger для фронтендера: `http://127.0.0.1:8080/docs`,
схема: `http://127.0.0.1:8080/openapi.json`.

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `WINE_CSV` | `strapi_output0709.csv` | Каталог (GT каталоговых полей) |
| `WINE_MEDIA_LINKS` | `artifacts/csv-media-links.jsonl` | **Кандидатные** связи slug↔файл (не verified) |
| `WINE_UPLOADS` | `prod-svoe-vino-strapi/.../uploads` | Каталог, из которого раздаются медиа |
| `WINE_MEDIA_BASE_URL` | — | Если картинки отдаёт CDN/фронт, а не backend |
| `WINE_CORS_ORIGINS` | `localhost:5173,3000`, `127.0.0.1:5173,3000` | Origin'ы фронтенда (`*` — все) |
| `WINE_TOKEN` | — | Если задан — Bearer на POST-эндпоинтах |
| `WINE_MAX_WORKERS` | `8` | На FastAPI — лимит всех одновременных POST **до чтения тела**, плюс лимит inference; переполнение → `503 busy`. Один API-процесс |
| `WINE_MAX_CONCURRENT_OCR` | `2` | Локальный OCR semaphore; входит в fingerprint policy |
| `WINE_HOST` / `WINE_PORT` | `127.0.0.1` / `8080` | Адрес uvicorn |
| `WINE_REQUEST_TIMEOUT` | `8.5` | Общий бюджет запроса, сек; входит в fingerprint policy (жёсткий SLA ещё не доказан) |
| `WINE_ROI_MODE` | `refuse` | `refuse` \| `center_80_crop` \| `full_frame_experimental` \| `auto_bbox_experimental` |
| `WINE_POLICY` | — | Текстовая policy v3 или fusion policy v4 с полным fingerprint; несовместимая/legacy policy прерывает запуск |
| `WINE_RELEASE_MANIFEST` | — | Опциональный локальный manifest из `wineid.release_bundle`; проверка checksum и путей до загрузки моделей; **не** подпись/подтверждение лицензий |
| `WINE_OCR_PROVIDER` | `none` | `none|tesseract|paddle|mineru|mistral`; `mineru` здесь — **локальный CLI для ROI**, не облачный Precision. Только явный `mistral` отправляет ROI вовне |
| `WINE_OCR_MODE` | `cold` | `warm` только для Paddle (рекомендуемый локальный онлайн-профиль после замеров); cold Paddle/MinerU — экспериментальный subprocess на запрос |
| `WINE_OCR_PYTHON` | API Python | Python OCR venv с `wineid` в `PYTHONPATH`, PaddleOCR/PaddlePaddle |
| `WINE_PADDLE_DET_DIR` / `WINE_PADDLE_REC_DIR` / `WINE_OCR_MANIFEST` | — | Обязательны для warm Paddle: локальные веса и manifest с SHA-256 всех файлов и версиями пакетов |
| `WINE_OCR_STARTUP_TIMEOUT` | `120` | Секунды для реального прогрева локального воркера при старте (не request timeout) |
| `WINE_TESSERACT` / `WINE_TESSDATA_DIR` | `tesseract` / — | CLI и установленный pinned `rus+eng` traineddata для локального baseline |
| `MISTRAL_API_KEY` | — | Только для явного `WINE_OCR_PROVIDER=mistral`, readiness remote OCR не подтверждается ключом |
| `WINE_CLASS_INDEX` | — | NPZ `clip_zero_shot build-classes --csv ...`, привязан к SHA каталога |
| `WINE_GALLERY` | — | NPZ `clip_eval build` ТОЛЬКО по вручную verified gallery manifest |
| `WINE_ADAPTER` | — | LoRA vision checkpoint; class index/gallery должны быть построены с ним же |
| `WINE_ALLOW_REMOTE_CODE` | — | `1` явно разрешает загрузку Jina/HF remote code после проверки лицензии |
| `WINE_BLEND_CONFIG` | — | JSON для `BlendConfig`; позволяет OCR-only blend без CLIP |
| `WINE_CONDITIONAL_OCR` | — | `1` включает **экспериментальную** эвристику пропуска OCR (без validation не включать) |
| `WINE_DEVICE` | — | torch device для Jina |
| `WINE_QUERY_VIEWS` | `1` | `3` — экспериментальная TTA: полный кадр + центр + низ этикетки, max-cosine; проверять на val |
| `WINE_VISION_STARTUP_TIMEOUT` | `120` | Секунды на загрузку и canary CLIP worker при старте/фоновом восстановлении; не входит в бюджет запроса |
| `WINE_ALIASES` | `producer_aliases.txt` | Проверенные алиасы производителей |

`.env` не читается автоматически; секреты — только через окружение. Для облачного **MinerU Precision (PDF/сканы)** используйте независимый opt-in CLI из [MINERU_CLOUD.md](MINERU_CLOUD.md); он никогда не запускается из wine HTTP и не подменяет `WINE_OCR_PROVIDER=mineru`.

## Эндпоинты

### `GET /api/health` (открыт)
```json
{"ready": true, "catalog_ready": true, "recognition_ready": false,
 "acceptance_ready": false, "catalog_sha256": "12a1…", "wine_count": 2103,
 "ocr_configured": false, "roi_mode": "refuse", "policy_configured": false,
 "search_backend": "fuzzywuzzy-0.18 difflib.SequenceMatcher", "media_configured": false}
```
`ready` = каталог загружен и его SHA совпадает с индексом. Это **не**
`recognition_ready`. Отдельный `ocr_ready` истинный только после реального canary локального OCR (у warm worker — если процесс ещё жив); `recognition_ready` требует OCR-ready + включённый ROI или визуальную ветку. Для Mistral доступность upstream не определяется ключом. `acceptance_ready` отдельно проверяет совместимость policy с текущим runtime и доступность необходимых веток; `policy_configured` означает только наличие policy. `GET /api/health/recognition` возвращает `503` при `recognition_ready=false`. Legacy `GET /ready` по-прежнему возвращает `503` только если не готов каталог. CLIP для HTTP запускается в одном отдельном persistent process: стартовый canary перед приёмом трафика; при зависании процессная группа убивается, `recognition_ready=false` до фонового восстановления (до трёх попыток с задержкой). Во время восстановления запросы не загружают веса и получают отказ; ответ `503 vision_unavailable` при отсутствии другого evidence, `503 busy` при занятости. Один процесс API означает один CLIP worker; межпроцессного лимита нет. Наличие живого процесса не гарантирует его отзывчивость до завершения текущего inference.

### `GET /api/wines?q=&limit=&offset=`
Каталог/автокомплит. `limit` 1–200. Без `q` — срез каталога, с `q` — ранжирование
текстовой веткой. Ответ: `{total, count, limit, offset, items:[WineCard]}`.

```json
{"total": 2103, "count": 1, "limit": 20, "offset": 0, "items": [{
  "slug": "vina-arpachina-arpachino-inohodets-sibirkovyy-beloe-ekstra-bryut-125",
  "name": "Вино. Арпачина", "producer": "Вино. Арпачина",
  "color": "Белое", "vintage": "", "grape": "сибирьковый",
  "category": "Белое", "sweetness": "suhoe",
  "media_url": "/api/media/….webp", "media_mapping_status": "candidate"}]}
```

### `GET /api/wines/{slug}`
Одна `WineCard`. `404 not_found`.

### `POST /api/search` (Bearer, если задан `WINE_TOKEN`)
Текстовая ветка (работает полностью офлайн, без OCR/весов).
```json
// запрос
{"text": "Арпачино Иноходец Сибирьковый", "k": 5, "vintage_mode": "soft"}
// ответ
{"text": "...", "backend": "…", "candidates": [
  {"wine_id": "…", "slug": "…", "rank": 1, "score": 94.26,
   "fuzzy_score": 100, "edit_distance": 3, "normalized_distance": 0.05,
   "match": "сибирьковый", "wine": { …WineCard… }}]}
```
`score` — ранжирующий балл [0,100], **не вероятность**. Тело `/api/search` ограничено 16 KiB, поле `text` — 4096 символов.

### `POST /api/recognize` (multipart, Bearer)
Поля: `image` (файл), опционально `bbox` = `"x0,y0,x1,y1"` (явный ручной ROI,
для отладки; без него действует `WINE_ROI_MODE`). Опциональный `roi_mode`
меняет путь только на запросе. **При настроенной policy любой bbox, override
ROI или ручная транскрипция не дают `accepted`** (`uncalibrated_request_profile`);
это отдельная диагностическая абляция, а не калиброванный профиль.

```json
{"status": "ambiguous", "slug": null,
 "reason_codes": ["policy_not_calibrated"],
 "versions": {"csv_sha256": "…", "search_backend": "…", "policy": null, "crop": "refuse"},
 "timings_ms": {"total": 41},
 "candidates": [ …как в /api/search, плюс `wine`… ],
 "ocr": {"roi_id": "…", "status": "ok", "raw_text": "…", "normalized_text": "…",
         "model": "mistral-ocr-latest", "latency_ms": 812,
         "error_code": null, "boxes": null, "confidence": null},
 "roi": {"roi_id": "…", "source_sha256": "…", "source_size": [w,h],
         "bbox": [x0,y0,x1,y1], "status": "manual|fallback_bbox",
         "crop_version": "…"}}
```

`status`: `accepted` (нужна калиброванная `WINE_POLICY`) · `ambiguous` ·
`unknown` · `unreadable` · `ambiguous_scene` · `error`. `slug` — точный slug CSV
только при `accepted`, иначе `null`.

Типовые `reason_codes`: `automatic_roi_unavailable` (режим `refuse`),
`ocr_not_configured`, `no_text`, `policy_not_calibrated`, `close_candidates`,
`insufficient_text_match`, `ocr_capacity_exceeded`, `vision_capacity_exceeded`, `deadline`, `busy`,
`invalid_size`, `invalid_bbox`, `multiple_label_bboxes`, `unauthorized`,
`uncalibrated_request_profile`, `policy_configuration_changed`, `partial_evidence`,
`vision_worker_failed`, `vision_worker_unavailable`.

Legacy/совместимость: `POST /v1/recognize` (то же тело). `POST /v1/eval/predict`
для **валидного** изображения принудительно отдаёт `{"slug": "<slug>"}` без
production policy, даже если основной API вернул `ambiguous`. Если ни один
источник не сконфигурирован, детерминированный первый slug каталога возвращается как
**технический fallback**, не распознавание (reason
`eval_catalog_fallback_no_evidence` в `Pipeline.predict(force_top1=True)`).
При падении/зависании CLIP fallback **запрещён** и eval возвращает `503`;
при частичном сбое с OCR evidence policy не принимает slug. Невалидное изображение →
`422` и error-JSON с `slug:null`, таймаут/пустой каталог → `503`.
При `WINE_ROI_MODE=refuse` только eval использует экспериментальный центральный
кроп для OCR. Это не детектор этикетки.

### `GET /api/media/{filename}`
Раздаёт **только** файлы из кандидатного артефакта (`image_path`), иначе `404`.

### `GET /api/catalog/collisions`
Диагностика неразличимости: `{groups, affected_slugs, class_fraction, largest_group}`.
Внутри группы точный Top-1 не обоснован.

## Опциональный online-бленд (CPU/offline тесты не требуют весов)

```bash
# После проверки лицензии CC BY-NC 4.0 и HF remote code:
python -m wineid.clip_zero_shot build-classes artifacts/classes.npz \
  --csv strapi_output0709.csv --prompt canonical \
  --adapter artifacts/lora/jina-clip-v2-lora-synthetic.pt --allow-remote-code
# Только при наличии ПОДТВЕРЖДЁННОЙ вручную gallery:
python -m wineid.clip_eval build verified_gallery.jsonl artifacts/gallery.npz \
  --adapter artifacts/lora/jina-clip-v2-lora-synthetic.pt --allow-remote-code
# Опционально при пересборке gallery: --synthetic-views 3 (fixed-seed, НЕ реальные фото)
WINE_CLASS_INDEX=artifacts/classes.npz WINE_ADAPTER=artifacts/lora/jina-clip-v2-lora-synthetic.pt \
  WINE_ALLOW_REMOTE_CODE=1 WINE_ROI_MODE=center_80_crop python -m wineid.api
```

`WINE_GALLERY=artifacts/gallery.npz` можно добавить только для gallery из
верифицированного manifest, совпадающего с каталогом/encoder/адаптером. Если
LoRA не нужен, соберите **оба** индекса базовой моделью и не задавайте
`WINE_ADAPTER`. Нельзя подмешивать исходный `artifacts/tz/images.csv` как
verified media или считать старый `artifacts/eval-jina-classes.npz`
совместимым с адаптером. Startup прерывается при несовпадении версий.
Запрос с несколькими бутылками требует отдельной проверки ROI: max по TTA
может смешать бутылки. Автоматический кроп/условный OCR, TTA и синтетическая
gallery пока экспериментальные. Синтетические виды создаются только из
верифицированных референсов (с seed и provenance), не превращают кандидатные
связи из CSV в подтверждённые.

В blended-ответе `candidates` содержит top-20, `score` — только отображаемая
относительная шкала (top-1=100), `blend_score` — **не** вероятность;
`signals` хранит сырые оценки, ранги и provenance, `family_size` — размер серии,
`attribute_score` — бонус OCR. `margin` — разность score top-1/top-2 в сырой
шкале бленда. Только при совместимой `fusion-policy-v4` добавляется
`confidence={p_top1_given_pool,p_top5_given_pool,margin_given_pool}`: softmax
условен на найденном пуле, не оценивает отсутствие вина в каталоге. Без policy
основной API возвращает candidates, но `slug:null`.

Для offline анализа сохранённых observations: `python -m wineid.fusion_eval
queries.jsonl evidence.jsonl --csv strapi_output0709.csv --split validation`;
`--tune` и `--calibrate-temperature` допустимы только для validation-only
manifest. Файл evidence JSONL: `query_id`, `evidence` (список полей
`CandidateEvidence`), `ocr_text`, опционально `latency_ms`. Для проверки утечек
перед тестом передайте `--calibration-manifest`; GT и SHA изображений
проверяются отдельно от evidence. `--policy` загружает тот же runtime (`WINE_*`), **повторно запускает** OCR/CLIP
по каждому байту проверенного manifest и отклоняет сохранённые evidence, если
источники/профиль/текст отличаются; нельзя использовать manual bbox или oracle
transcript. Совпадать должны, в частности, `WINE_REQUEST_TIMEOUT` и параметры
`fusion_run --seconds` (по умолчанию 30 против серверных 8.5), ROI и индексы.
Внешние OCR-провайдеры с policy запрещены. Для acceptance на test требуется
`--calibration-manifest` с проверкой утечек. Старые evidence без image/catalog/config
SHA — только ranking-диагностика. SHA в JSONL **не подпись**, но повторная
инференция не доверяет вложенному SHA. Результат требует замороженного bundle. На `eval/`
параметры не подбирать. Policy
создаётся программно через `build_fusion_policy(..., wines=wines, config=pipeline.fusion_config, ...)`
из real validation с проверенными изображениями (и `gallery_manifest` при
использовании галереи); JSON = `{**policy.payload(), "artifact_sha256": policy.artifact_sha256}`.
Синтетическая validation полезна только для предварительного поиска весов.

## Проверенный локальный прогон на готовом LoRA (без независимого GT)

Инструкции, ограничения и воспроизводимые отчёты по настоящим фотографиям:
[plans/REAL_RUN.md](plans/REAL_RUN.md). `wineid.fusion_run` выдаёт сырые наблюдения
и диагностическое совпадение с **кандидатными**, не вручную подтверждёнными
связями uploads. На трёх фото `eval/` нет GT — их предсказания не показывают
качество. При наличии независимого размеченного manifest CLI выполняет аудит
до загрузки весов и запускает `fusion_eval`. Настоящий OCR требует отдельного
`WINE_OCR_PROVIDER=mistral` и `--ocr live --allow-live-ocr`; для локального
`--ocr live` нужен явный `WINE_OCR_PROVIDER=tesseract|paddle|mineru` и локальный
canary. Без выбранного провайдера сеть не используется.

## Локальный OCR → ROI → поиск (экспериментальный профиль)

```bash
# На целевом сервере в OCR venv: установите PaddleOCR/PaddlePaddle совместимых
# версий, предварительно загрузите PP-OCRv5 ru detector/recognizer в локальные
# каталоги; проверьте источник и лицензию конкретных весов.
PYTHONPATH=$PWD /path/to/ocr-venv/bin/python -m wineid.ocr_manifest \
  --det /models/det --rec /models/rec --output /models/ocr-manifest.json
# Сделайте каталоги/manifest read-only, закрепите digest выпуска, запретите egress.
WINE_OCR_PROVIDER=paddle WINE_OCR_MODE=warm \
  WINE_OCR_PYTHON=/path/to/ocr-venv/bin/python \
  WINE_PADDLE_DET_DIR=/models/det WINE_PADDLE_REC_DIR=/models/rec \
  WINE_OCR_MANIFEST=/models/ocr-manifest.json WINE_ROI_MODE=center_80_crop \
  WINE_MAX_WORKERS=1 PYTHONPATH=$PWD python -m wineid.api
# Или CPU canary: WINE_OCR_PROVIDER=tesseract WINE_ROI_MODE=center_80_crop python -m wineid.api
```

Для startup двух HTTP адаптеров используется единый `wineid.runtime.open_runtime`:
конфигурация проверяется до загрузки моделей, OCR закрывается при ошибке и
после shutdown. С `WINE_POLICY` требуются pinned локальные OCR веса:
Paddle warm + manifest либо Tesseract с явным `WINE_TESSDATA_DIR` (rus+eng).
Mock, MinerU и Mistral с policy в HTTP не запускаются; без policy доступны
как диагностические источники. В fingerprint text v3 / fusion v4 входят код решений, версии
Python/библиотек, алиасы, ROI, OCR, индексы/encoder, способ выполнения CLIP,
бленд, request timeout, лимит OCR и conditional OCR. Старые fusion v3 не переносить
переименованием: калибровать заново на независимой validation. При работе CLIP
весов и remote code в HF-кэше нет — startup завершится ошибкой (egress в child
отключён). Смена настройки требует новой policy. Это ещё **не** подтверждённый
выпуск или wall-clock SLA: декодирование/PNG сериализация и скоринг в parent,
синхронный OCR и совместная нагрузка могут превышать бюджет.

Один локальный worker на один процесс API, очередь OCR не накапливается: занятость → `503 busy` без визуальной альтернативы. При истечении deadline зависший warm worker и его потомки убиваются; проверка SHA весов и canary повторяются **в фоновом восстановлении** (до 3 попыток с задержкой и cooldown после неудач), а не в следующем HTTP-запросе. До восстановления `ocr_ready=false`, без другой evidence запрос получит `503 ocr_unavailable`; изменение/порча manifest или весов блокирует восстановление до перезапуска с проверенным bundle. Forced eval не подставляет каталоговый slug при отказе OCR worker. Cold OCR передаёт подготовленный ROI PNG движку без повторного декодирования и сжатия в parent; препроцессинг в `roi.py` по-прежнему синхронный. Несколько uvicorn-процессов создают несколько копий модели — **не включать** без общего лимита CPU/GPU. `center_80_crop` — только явный fallback, не локализация этикетки. На запросе веса не скачиваются в warm режиме; тест на target OS/GPU и независимая калибровка всё ещё обязательны. `WINE_ROI_MODE=refuse` не вызывает OCR без bbox.

Для независимой диагностической оценки (все split проверяются до запуска OCR):
`WINE_OCR_PROVIDER=tesseract python -m wineid.ocr_eval private-queries.jsonl --split validation --output private-ocr.jsonl`.
JSONL требует `query_id,image_path,image_sha256,source_group,split,bbox` (координаты после EXIF), `transcription` (буквальный текст), `gt_status,gt_slug,gt_source`. Отчёт на stdout содержит только суммарные CER/WER (NFKC, lowercase, ё→е, пробелы; пустые/ошибочные результаты входят в знаменатель), Top-5 по OCR и ручному тексту, задержки; raw-тексты пишутся **только** в новый private-файл с правами 0600 при `--output`. Не использовать три неразмеченных `eval/` как GT. Опциональный реальный CLI smoke: `WINE_SMOKE_OCR=1 python -m pytest tests/test_ocr_eval.py -q` (понадобятся установленные `rus+eng`).

## Подготовка локального inventory выпуска (PR-06, ещё не RC)

После копирования утверждённых файлов в приватный read-only каталог можно
зафиксировать их байты; роль `lock-api`, `lock-ocr` и т.п. — отдельные
подготовленные lock-файлы (текущий `constraints.txt` не является полным lock):

```bash
python -m wineid.release_bundle build /private/bundle /private/bundle/files.json \
  --revision "$(git rev-parse HEAD)" \
  --file csv=strapi_output0709.csv --file aliases=producer_aliases.txt \
  --file lock-api=api.lock  # добавить policy/classes/gallery/adapter/OCR роли по выбранному профилю
python -m wineid.release_bundle verify /private/bundle /private/bundle/files.json
WINE_RELEASE_MANIFEST=/private/bundle/files.json WINE_CSV=/private/bundle/strapi_output0709.csv \
  WINE_ALIASES=/private/bundle/producer_aliases.txt python -m wineid.api
```

Все runtime-артефакты (`WINE_POLICY`, `WINE_CLASS_INDEX`, `WINE_GALLERY`,
`WINE_ADAPTER`, `WINE_OCR_MANIFEST`) при заданном manifest должны быть перечислены
соответствующими ролями и путями внутри каталога manifest. Проверка не заменяет
подпись manifest, сквозной lock HF/torch, проверку лицензий, pinned system image,
реальный canary, нагрузку и rollback. При незакоммиченных изменениях SHA коммита
не описывает всю рабочую копию: собирать bundle только из чистого checkout.
Облачный MinerU использует **отдельный** `MINERU_CLOUD.md` и приватную очередь,
не включается в этот HTTP-профиль и не запускается по `WINE_RELEASE_MANIFEST`.

## Ошибки
Единый формат: `{"status": "error", "slug": null, "reason_codes": ["…"]}`.
`400` — невалидный ввод/дополнительные multipart-поля, `401` — нет/неверный Bearer (до чтения тела), `413` — размер тела (multipart до 12 MiB + 64 KiB, JSON search до 16 KiB), `408 deadline` — медленный upload в FastAPI,
`404` — не найдено, `500 internal` — без деталей, `503 busy` — перегрузка до чтения тела либо inference, `503 vision_unavailable` / `503 ocr_unavailable` — отказ соответствующего worker при отсутствии альтернативной evidence. Chunked upload принимается FastAPI с тем же лимитом; stdlib HTTP требует Content-Length и отвергает chunked. Совместимые лимиты reverse proxy, соединений и TLS требуют отдельной настройки. Дедлайн upload включён в оставшийся бюджет ROI/OCR/поиска; CLIP и warm Paddle inference изолированы и killable, но общий жёсткий wall-clock deadline для parent preprocessing, внешнего OCR и совместной нагрузки ещё не доказан.

## Что важно знать фронтенду (ограничения)
- **Фотосъёмка сейчас не распознаётся «из коробки»**: `WINE_ROI_MODE=refuse` и
  `ocr_configured:false` → `unreadable` в основном API без визуального индекса. Нужен явно выбранный проверенный локальный OCR и ROI (ручной bbox или экспериментальный автоматический режим) либо отдельный визуальный индекс; ключ Mistral сам по себе OCR не включает.
- **Работает офлайн уже сейчас**: каталог, карточки, текстовый поиск, диагностика.
- `media_url` — **кандидатная** связь (`media_mapping_status: "candidate"`), не
  verified CMS relation; не показывать как «эталон».
- Никакой ответ не является гарантией точного SKU; `candidates` — ранжирование.
- Backend stateless по пользовательским фото: загруженные изображения не
  сохраняются и не логируются.
