# Архитектура веб-фронтенда «Сканер вин»

Как устроен мобильный модуль в `web/`. Требования к экранам — в `docs/tz-mobile-web.md`, запуск — в `web/README.md`.

## 1. Слои

```
app/          страницы и маршруты
features/     scan, sommelier, suggest
components/   ui, блоки карточки, layout
services/     apis, типы, константы, аналитика
```

Зависимости идут только вниз. `components/` не ходит в API и router. Сеть вызывается через объект `apis` в `services/apis/index.ts`.

Обращения к `window`, `location`, `navigator` и `sessionStorage` живут в composables и обработчиках, а не при загрузке модуля. Исключение — `services/constants/env.ts`, он читает `import.meta.env`. Стили компонентов — `scoped`. Глобальны только `tokens.css` и `base.css`.

## 2. Каталоги

```
web/
├── .env.development
├── .env.example
├── public/bottles/       placeholder.svg, placeholder-white.svg
└── src/
    ├── app/              router, scan, wine, similar, error
    ├── components/       ui, wine, layout
    ├── composables/      useToast, useFocusTrap, useShare, useDebugPanel
    ├── features/
    │   ├── scan/         store, камера, подготовка фото, шторка, debug
    │   ├── sommelier/    блок, fallback, поиск похожих по вкусу
    │   └── suggest/      форма «Предложить вино»
    └── services/
        ├── apis/         http-client, live-scan, map-recognize, map-wine, scan-cache
        ├── types/        zod-схемы и выведенные типы
        └── analytics/    события в console.debug, логгер
```

Тесты лежат рядом с кодом в `__tests__/`.

Если у карточки пустой `media_url`, `map-wine.ts` подставляет `placeholder-white.svg` для белого вина и `placeholder.svg` для остальных.

## 3. Скан

`ScanPage` отдаёт файл в `useScan`. Store готовит JPEG (`useImagePreparer`), затем `apis.scan` вызывает `POST /api/recognize`. Заголовок `X-Client-Normalized: 1`. Таймаут — не меньше 10 с (`VITE_SCAN_TIMEOUT_MS` и нижняя граница в `timings.ts`).

Ответ `RecognizeOut` приводится к `ScanResult`:

| Статус бэкенда | Экран |
| --- | --- |
| `accepted` | карточка, `match.status = exact` |
| `ambiguous` | карточка top-1 и шторка, `uncertain` |
| `unknown` | похожие. Если кандидатов нет, но OCR отдал текст — дополнительный `POST /api/search` |
| `unreadable`, `ambiguous_scene` | повторная съёмка |
| `error` | экран ошибки. `reason_codes` с `deadline` — таймаут |

`scanId` создаёт браузер (`crypto.randomUUID`) и кладёт результат в `sessionStorage`, последние 12. `GET` скана с бэкенда нет: `apis.getScan` читает этот кэш. Фото туда не пишется.

При `413` store один раз сжимает фото до 960 px и повторяет запрос.

`WinePage` и `SimilarPage` берут `ScanResult` из store, а при прямом заходе — из кэша. Без `?scan=` скановые блоки не рисуются. Шторка `uncertain` открывается через 600 мс один раз на `scanId`.

Карточка по slug — `GET /api/wines/{slug}`. Ответ `WineCard` (slug, name, producer, color, vintage, grape, category, sweetness, media_url). Регион, описание, крепость, температура, блюда и рейтинги в этом ответе не приходят, соответствующие блоки скрываются.

## 4. Сеть

`http-client.ts` ходит через axios (`requestLive`): Bearer из `VITE_API_TOKEN`, таймаут, разбор ошибок бэкенда (`reason_codes`) и тела `{ error: { code } }` в `ApiError`. Коды: `NETWORK_OFFLINE`, `NETWORK_FAILED`, `TIMEOUT`, `ABORTED`, `SERVER_ERROR`, `INVALID_RESPONSE`, `IMAGE_TOO_LARGE`, `UNSUPPORTED_IMAGE`, `NOT_FOUND`, `UNKNOWN`.

Ответы recognize, search и wines проверяются zod-схемами. Невалидный JSON — `INVALID_RESPONSE`.

`apis.sommelier` не открывает поток: блок сомелье сразу берёт текст из `sommelier-fallback.ts`. Подбор «другого вкуса» дополнительно вызывает `POST /api/search`. Форма «Предложить вино» подтверждается на экране и в сеть не уходит.

Парсер SSE в `services/apis/sse.ts` в рантайме не вызывается.

Dev-сервер проксирует `/api` на `127.0.0.1:8080`.

## 5. Маршруты

| Маршрут | Экран | Параметры |
| --- | --- | --- |
| `/` | редирект на `/scan` | — |
| `/scan` | сканер, загружается сразу | `?source=camera\|gallery`, `?debug=1` |
| `/wines/:slug` | карточка, lazy | `?scan=`, `?debug=1` |
| `/scan/:scanId/similar` | похожие, lazy | `?debug=1` |

Остальные пути ведут на `/scan`. `?source=` только фокусирует кнопку: браузер не открывает камеру без жеста. `?debug=1` или `VITE_DEBUG_PANEL=true` показывает панель метрик. `score` там — балл ранжирования 0–100, делённый на 100.

`getWine` с 404 и `?scan=` ведёт на экран похожих. Без `scan` — на `/scan` с тостом. Если кэша скана нет, карточка открывается без скановых блоков, а экран похожих показывает ошибку с «Повторить».

## 6. Ошибки

| Ситуация | Реакция |
| --- | --- |
| Не изображение, файл больше 25 МБ, битый файл | тост, остаёмся на `/scan` |
| Сбой декодирования или ресайза | тост, store сброшен |
| 413 | один повтор с 960 px, иначе экран ошибки |
| 422, нечитаемая этикетка | экран повторной съёмки |
| Нет сети, 5xx, невалидный ответ, таймаут | экран ошибки, «Повторить» / «Снять заново» |
| Отмена | `AbortController`, без сообщения |
| Ошибка рендера | `App.vue`, экран «Что-то пошло не так» |

Тексты — в `services/constants/messages.ts`. Неожиданные ошибки пишет `services/analytics/logger.ts`.

«Сканировать» открывает живой видоискатель. Если камеры нет или доступ уже запрещён, в том же жесте открывается системная камера. Отказ в диалоге разрешения запоминается на вкладку без тоста.

## 7. Интерфейс

Токены — `assets/tokens.css`. Заголовки — локальный Playfair Display 600 (`assets/fonts`, `font-display: swap`), текст — системный sans. Кнопки, чипы, скелетон, шторка и тосты собраны в `components/ui` без UI-библиотек. Зона касания от 44 px, `prefers-reduced-motion` отключает анимации. Макет рассчитан на 360–430 px, на десктопе колонка 480 px.

## 8. События и тесты

`track` в `services/analytics/events.ts` пишет событие в `console.debug`. Запроса `/api/events` нет.

Тесты — Vitest и jsdom: подготовка фото, выбор экрана, схемы, причины, маппинг recognize и карточки, кэш скана, сомелье, шторка, экран похожих, видоискатель. Ресайз через canvas и EXIF на устройстве проверяются вручную. E2E нет.

## 9. Решения

- Скан и карточка ходят в живой API. Отдельного мок-транспорта нет.
- `ScanPage` в начальном чанке, карточка и похожие — lazy. `vue` и `zod` вынесены в отдельные чанки.
- Оверлей обработки и экран ошибки скана живут в `App.vue`, чтобы пережить смену маршрута.
- Повтор при 413 — один, с длинной стороной 960 px.
- Сомелье собирает ответ на клиенте из карточки и, для другого вкуса, из `POST /api/search`.
- Последние сканы хранят slug, scanId и название, без фото.
- `?source=` не открывает диалог само, только фокусирует кнопку.
- Шрифт лежит локально, внешних запросов за ним нет.
