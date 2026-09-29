# Инструкция для фронтендера: как запустить и подключить бэкенд

Привет! Бэкенд написан на **FastAPI** и готов к интеграции.  
Вся интерактивная документация и песочница запросов (Swagger UI) доступна сразу после запуска:
👉 **`http://127.0.0.1:8080/docs`**  
OpenAPI JSON схема: `http://127.0.0.1:8080/openapi.json`

---

## 1. Быстрый старт бэкенда (3 команды)

В корне репозитория:

```bash
# 1. Установить зависимости бэкенда
pip install -r requirements-api.txt

# 2. Запустить сервер
python -m wineid.api
```

Сервер поднимется на `http://127.0.0.1:8080`.  
CORS по умолчанию уже открыт для стандартных портов фронтенда:
- `http://localhost:5173` (Vite)
- `http://localhost:3000` (Next.js / CRA / Nuxt)
- `http://127.0.0.1:5173`, `http://127.0.0.1:3000`

Если твой фронт на другом порту, просто передай переменную:
```bash
WINE_CORS_ORIGINS="http://localhost:4000,http://localhost:8081" python -m wineid.api
# или для тестов разрешить все:
WINE_CORS_ORIGINS="*" python -m wineid.api
```

---

## 2. Как устроен бэкенд и данные

1. **Каталог вин** загружается из `strapi_output0709.csv` (2103 уникальных SKU вин).
2. **Текстовый поиск и просмотр каталога работают полностью офлайн** прямо сейчас, без видеокарт и без внешних ключей.
3. **Распознавание по фото**: в базовом режиме сервер возвращает статус `unreadable` (`automatic_roi_unavailable`), потому что по правилам проекта автоматический локализатор этикеток по умолчанию выключен. Для отладки интерфейса можно передавать параметр `bbox` или запустить сервер с тестовым режимом `WINE_ROI_MODE=full_frame_experimental`.

---

## 3. Основные ручки (REST API)

### 3.1. Проверка состояния: `GET /api/health`
Вызывай при старте приложения, чтобы показать статус сервиса.

**Ответ (200 OK):**
```json
{
  "ready": true,
  "catalog_ready": true,
  "recognition_ready": false,
  "catalog_sha256": "12a1b0b6...",
  "wine_count": 2103,
  "ocr_configured": false,
  "roi_mode": "refuse",
  "policy_configured": false,
  "search_backend": "fuzzywuzzy (installed ratio backend)",
  "media_configured": false
}
```
* `ready` / `catalog_ready: true` — каталог загружен, поиск и список вин доступны.
* `recognition_ready: false` — фото-поиск пока на паузе (нужен OCR/веса).

---

### 3.2. Список вин и автокомплит: `GET /api/wines`
Пагинация + живой поиск по каталогу.

**Параметры query:**
- `limit` (int, default 20, max 200)
- `offset` (int, default 0)
- `q` (string, опционально) — поисковая фраза (название, винодельня, сорт).

**Пример:** `GET /api/wines?limit=10&offset=0&q=рислинг`

**Ответ:**
```json
{
  "total": 2103,
  "count": 10,
  "limit": 10,
  "offset": 0,
  "items": [
    {
      "slug": "fanagoriya-avtorskiy-stil-risling-beloe-suhoe-125",
      "name": "Авторский стиль. Рислинг",
      "producer": "Фанагория",
      "color": "Белое",
      "vintage": "",
      "grape": "рислинг",
      "category": "Белое",
      "sweetness": "suhoe",
      "media_url": "/api/media/fanagoriya_...webp",
      "media_mapping_status": "candidate"
    }
  ]
}
```
* Поле `media_url`: если `null`, картинка пока не привязана. Если строка — картинку можно вставлять в `<img src="http://127.0.0.1:8080" + wine.media_url />`.
* `media_mapping_status`: `"candidate"` (найдена по имени файла) или `"unresolved"`.

---

### 3.3. Карточка одного вина: `GET /api/wines/{slug}`
Для страницы детального просмотра вина.

**Пример:** `GET /api/wines/aligote-barrel-2024`

**Ответ (200 OK):**
```json
{
  "slug": "aligote-barrel-2024",
  "name": "Алиготе Баррель, 2024",
  "producer": "Коммуналка",
  "color": "Белое",
  "vintage": "2024",
  "grape": "алиготе",
  "category": "Белое",
  "sweetness": "suhoe",
  "media_url": null,
  "media_mapping_status": "unresolved"
}
```
Если вина нет — `404` с телом `{"status": "error", "slug": null, "reason_codes": ["not_found"]}`.

---

### 3.4. Умный поиск: `POST /api/search`
Быстрый поиск с ранжированием (IDF + лексический блендер + учет опечаток и сортов).

**Тело запроса (JSON):**
```json
{
  "text": "Арпачино Иноходец Сибирьковый",
  "k": 5,
  "vintage_mode": "soft"
}
```
* `k` — сколько кандидатов вернуть (1–100).
* `vintage_mode`: `"soft"` (учитывает год мягко) или `"hard"` (отсекает неподходящие года).

**Ответ:**
```json
{
  "text": "Арпачино Иноходец Сибирьковый",
  "backend": "fuzzywuzzy (installed ratio backend)",
  "candidates": [
    {
      "wine_id": "vina-arpachina-arpachino-inohodets-sibirkovyy-beloe-ekstra-bryut-125",
      "slug": "vina-arpachina-arpachino-inohodets-sibirkovyy-beloe-ekstra-bryut-125",
      "rank": 1,
      "score": 94.26,
      "fuzzy_score": 100,
      "edit_distance": 3,
      "normalized_distance": 0.05,
      "match": "сибирьковый",
      "wine": {
        "slug": "vina-arpachina-arpachino-inohodets-sibirkovyy-beloe-ekstra-bryut-125",
        "name": "Вино. Арпачина",
        "producer": "Вино. Арпачина",
        "color": "Белое",
        "vintage": "",
        "grape": "сибирьковый",
        "category": "Белое",
        "sweetness": "suhoe",
        "media_url": "/api/media/vina_arpachina_...webp",
        "media_mapping_status": "candidate"
      }
    }
  ]
}
```
`score` — балл совпадения (0..100). **Это не процент вероятности**, а сравнительный ранг.

---

### 3.5. Распознавание по фотографии: `POST /api/recognize`
Загрузка фотографии этикетки / бутылки.

**Формат запроса:** `multipart/form-data`
- Поле `image` — файл изображения (JPEG, PNG, WEBP, до 12 МБ).
- Поле `bbox` (string, опционально) — ручные координаты этикетки `"x0,y0,x1,y1"`.

**Пример отправки на JS/TS:**
```typescript
const formData = new FormData();
formData.append('image', fileInput.files[0]);
// опционально, если на фронте есть кроппер:
// formData.append('bbox', '120,340,650,980');

const res = await fetch('http://127.0.0.1:8080/api/recognize', {
  method: 'POST',
  body: formData
});
const data = await res.json();
```

**Формат ответа:**
```json
{
  "status": "ambiguous",
  "slug": null,
  "reason_codes": ["policy_not_calibrated"],
  "versions": {
    "csv_sha256": "12a1...",
    "search_backend": "fuzzywuzzy...",
    "policy": null,
    "crop": "oriented-rgb-bbox-v1"
  },
  "timings_ms": { "total": 84 },
  "candidates": [ ...список кандидатов как в /api/search... ],
  "ocr": {
    "roi_id": "manual-0",
    "status": "ok",
    "raw_text": "Рислинг 2021 Фанагория",
    "normalized_text": "рислинг 2021 фанагория"
  },
  "roi": {
    "bbox": [120, 340, 650, 980],
    "status": "manual"
  }
}
```

**Возможные статусы (`status`):**
- `accepted`: вино надежно определено. В поле `slug` будет точный идентификатор вина.
- `ambiguous`: найдено несколько близких кандидатов (см. массив `candidates`). В интерфейсе стоит показать топ-3 или топ-5 карточек пользователю на выбор.
- `unreadable`: на фото не удалось выделить этикетку или прочесть текст (код причины в `reason_codes`).
- `unknown`: текст прочитан, но такого вина нет в каталоге.
- `error`: ошибка сервиса или превышение тайм-аута (код в `reason_codes`).

---

### 3.6. Раздача картинок: `GET /api/media/{filename}`
Статическая отдача файлов этикеток, привязанных к каталогу. Отдает файлы только из проверенного белого списка каталога.

---

## 4. Стандартный формат ошибок
Все ошибки возвращаются в единой структуре:

```json
{
  "status": "error",
  "slug": null,
  "reason_codes": ["not_found"]
}
```

Коды HTTP:
- `400` — некорректный запрос (неверный формат bbox, пустой файл, некорректная пагинация).
- `401` — если на сервере включен `WINE_TOKEN`, но заголовок `Authorization: Bearer <token>` не передан.
- `404` — вино или изображение не найдено (`["not_found"]`).
- `413` — слишком большой файл (`["invalid_size"]` или `["invalid_body_size"]`).
- `503` — сервер занят (`["busy"]`), слишком много параллельных запросов распознавания.

---

## 5. Типичный сценарий интерфейса (User Flow)

1. **Главная страница:**
   - Поле поиска (input) ➔ на каждое изменение текста запрос `GET /api/wines?q={text}&limit=10` для выпадающего списка подсказок.
2. **Экран каталога:**
   - Сетка карточек ➔ `GET /api/wines?limit=24&offset={page * 24}`.
   - Клик по карточке ➔ переход на `/wine/{slug}` и запрос `GET /api/wines/{slug}`.
3. **Сканирование этикетки (камера/фото):**
   - Пользователь загружает фото ➔ `POST /api/recognize`.
   - Если `status === "accepted"` ➔ сразу открываем карточку `data.slug`.
   - Если `status === "ambiguous"` ➔ выводим блок: *«Мы нашли несколько похожих вин, выберите ваше:»* и рендерим карточки из `data.candidates`.
   - Если `status === "unreadable"` ➔ сообщение: *«Не удалось четко распознать этикетку, попробуйте сфотографировать ближе или введите название поиском»*.

---

Если понадобятся дополнительные поля, сортировки или фильтры по цвету/году — пиши бэкендеру, доработаем!
