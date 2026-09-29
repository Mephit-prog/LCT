# API интеграции (FastAPI backend)

**Статус: экспериментальный исследовательский прототип.** Это тонкий HTTP-слой
поверх существующих `wineid.pipeline.Pipeline` и `wineid.search.Index`. Семантика
распознавания не меняется: по умолчанию пайплайн **отказывается** от
автоматического распознавания, `score`/cosine — не вероятности, независимого
GT/метрик качества нет. Backend не «делает вид», что умеет распознавать.

## Запуск

```bash
python3 -m pip install -r requirements-api.txt
python3 -m wineid.api           # uvicorn, по умолчанию 127.0.0.1:8080
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
| `WINE_MAX_WORKERS` | `8` | Лимит одновременных распознаваний (иначе `503 busy`) |
| `WINE_HOST` / `WINE_PORT` | `127.0.0.1` / `8080` | Адрес uvicorn |
| `WINE_REQUEST_TIMEOUT` | `8.5` | Общий дедлайн запроса, сек |
| `WINE_ROI_MODE` | `refuse` | `refuse` \| `full_frame_experimental` \| `auto_bbox_experimental` |
| `WINE_POLICY` | — | Путь к калиброванной policy (иначе `ambiguous`) |
| `MISTRAL_API_KEY` | — | Без ключа OCR не вызывается (`ocr_configured:false`) |

`.env` не читается автоматически; секреты — только через окружение.

## Эндпоинты

### `GET /api/health` (открыт)
```json
{"ready": true, "catalog_sha256": "12a1…", "wine_count": 2103,
 "ocr_configured": false, "roi_mode": "refuse", "policy_configured": false,
 "search_backend": "fuzzywuzzy (installed ratio backend)", "media_configured": false}
```
`ready` = каталог загружен и его SHA совпадает с индексом. Это **не**
`recognition_ready`. Legacy: `GET /ready` (то же тело, 503 если не ready).

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
`score` — ранжирующий балл [0,100], **не вероятность**.

### `POST /api/recognize` (multipart, Bearer)
Поля: `image` (файл), опционально `bbox` = `"x0,y0,x1,y1"` (явный ручной ROI,
для отладки; без него действует `WINE_ROI_MODE`).

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
`insufficient_text_match`, `ocr_capacity_exceeded`, `deadline`, `busy`,
`invalid_size`, `invalid_bbox`, `multiple_label_bboxes`, `unauthorized`.

Legacy/совместимость: `POST /v1/recognize` (то же тело) и
`POST /v1/eval/predict` → `{"slug": "<slug>"|null}` для `eval/participant_test.sh`.

### `GET /api/media/{filename}`
Раздаёт **только** файлы из кандидатного артефакта (`image_path`), иначе `404`.

### `GET /api/catalog/collisions`
Диагностика неразличимости: `{groups, affected_slugs, class_fraction, largest_group}`.
Внутри группы точный Top-1 не обоснован.

## Ошибки
Единый формат: `{"status": "error", "slug": null, "reason_codes": ["…"]}`.
`400` — невалидный ввод, `401` — нет/неверный Bearer, `413` — размер тела,
`404` — не найдено, `500 internal` — без деталей, `503 busy` — перегрузка.

## Что важно знать фронтенду (ограничения)
- **Фотосъёмка сейчас не распознаётся «из коробки»**: `WINE_ROI_MODE=refuse` и
  `ocr_configured:false` → `unreadable`. Нужны разрешённый OCR-ключ и/или
  экспериментальный ROI (`auto_bbox_experimental` + `requirements-vision.txt`).
- **Работает офлайн уже сейчас**: каталог, карточки, текстовый поиск, диагностика.
- `media_url` — **кандидатная** связь (`media_mapping_status: "candidate"`), не
  verified CMS relation; не показывать как «эталон».
- Никакой ответ не является гарантией точного SKU; `candidates` — ранжирование.
- Backend stateless по пользовательским фото: загруженные изображения не
  сохраняются и не логируются.
