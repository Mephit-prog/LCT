# Код-ревью: полный обход `wineid/`, `tests/` и конфигурации

**Дата:** 2026-09-25
**Охват:** 21 модуль `wineid/*.py`, 9 файлов `tests/*.py`, `requirements*.txt`, скрипты `eval/`, сверка с `IMPLEMENTATION.md`, `plans/STATUS.md`, `JINA_CLIP.md`, `tz-clip-text-ranking.md`, `tz-wine-label-retrieval.md`.
**Метод:** ручное чтение всех модулей и тестов; запуск `python3 -m pytest -q` (**65 passed**, 1.91 s), `python3 -m compileall -q wineid tests` (OK), `pyflakes wineid tests`.

## Вердикт

Кодовая база — **аккуратный исследовательский прототип с сильной defensive-культурой**: абстенция вместо угадывания, аудиты provenance/leakage до выбора split, атомарная публикация индексов, явные consent-флаги для весов. Критичных (P0) дефектов, ломающих существующий функционал или открывающих эксплуатационную дыру в текущей конфигурации (loopback, без OCR по умолчанию), **не выявлено**.

Основные проблемы сосредоточены в новых ТЗ-модулях ([`wineid/train_clip.py`](wineid/train_clip.py), [`wineid/catalog_export.py`](wineid/catalog_export.py)) — **документация и код расходятся в трёх местах**, из-за чего заявленный пайплайн обучения не выполним «из коробки», и в HTTP-слое для будущего продакшена (отсутствие аутентификации, неограниченные потоки). Плюс несколько P2-замечаний по чистоте кода и пробелам покрытия.

| Severity | Кол-во | Кратко |
|---|---|---|
| P0 | 0 | Не выявлено |
| P1 | 6 | См. таблицу ниже |
| P2 | 12 | См. таблицу ниже |

---

## P1 — важно (исправить до использования соответствующего пути)

| # | Модуль | Проблема |
|---|---|---|
| 1 | [`wineid/train_clip.py`](wineid/train_clip.py:274) | **ТЗ §6.3 «тот же производитель ×2» не применяется в тренировке.** Реализовано только в numpy-эталонной [`info_nce_loss()`](wineid/train_clip.py:91); torch-путь [`clip_loss()`](wineid/train_clip.py:320) — обычный симметричный InfoNCE без весов. В [`train()`](wineid/train_clip.py:271) вычисляется `producers` и не используется (подтверждено pyflakes: `local variable 'producers' is assigned to but never used`). Докстринг и `IMPLEMENTATION.md` утверждают обратное. |
| 2 | [`wineid/catalog_export.py`](wineid/catalog_export.py:107) + [`wineid/train_clip.py`](wineid/train_clip.py:206) | **Разрыв контракта пути: обучение не найдёт изображения.** Exporter пишет в `images.csv` голый basename (`image_path = original.name`, см. [`wineid/catalog_media.py`](wineid/catalog_media.py:111)), а [`load_image_bytes()`](wineid/train_clip.py:181) резолвит `path` относительно директории `images.csv` (`root = Path(images_csv).resolve().parent`). При стандартном выводе `--out` в отдельную папку все файлы `FileNotFoundError`. Нужен либо `path` относительно `uploads`, либо явный параметр `--images-root`. |
| 3 | [`wineid/train_clip.py`](wineid/train_clip.py:299) | **Обученный адаптер нельзя применить: нет загрузчика/инференс-пути.** `torch.save(best_state)` пишет состояние, но ни один модуль не умеет его загружать и применять LoRA-веса к модели (нет аналога `attach_lora` для inference, нет `merge`/`load`). Результат обучения не потребляем. |
| 4 | [`wineid/server.py`](wineid/server.py:96) | **Отсутствуют аутентификация и ограничение потоков.** `ThreadingHTTPServer` плодит поток на запрос без лимита; `/v1/recognize` отдаёт полный результат (OCR-текст, кандидаты, timing). На loopback и при «refuse»-ROI риск ограничен, но это прямо противоречит производственным целям (в `STATUS.md` B06/B10 уже открыты; здесь фиксирую как P1 для интеграции). Плюс нет ограничения времени чтения тела сверх `settimeout(8.5)` на соединение — приемлемо, но slow-body теста нет (см. P2-13). |
| 5 | [`wineid/train_clip.py`](wineid/train_clip.py:165) | **`attach_lora()` жёстко завязан на структуру HF CLIP** (`vision_model.encoder.layers[i].self_attn.{q,k,v,out}_proj`). Архитектура `jinaai/jina-clip-v2` (SigLIP-подобная vision) не проверялась — при несовпадении атрибутов код корректно падает с `RuntimeError`, но это значит, что заявленный «vision-LoRA» может быть невыполним без доработки маппинга. Требуется smoke-проверка на реальной модели до запуска обучения. |
| 6 | [`wineid/search.py`](wineid/search.py:27) | **Fallback-бэкенд не score-совместим, а fuzzywuzzy — GPL-2.0.** При отсутствии `fuzzywuzzy` ранжирование тихо меняется (difflib fallback, документировано, но результат не совместим с откалиброванным порогом). Для production нужен единый разрешённый бэкенд и перекалибровка policy при смене. Отмечено в `IMPLEMENTATION.md`, остаюсь на P1 из-за эксплуатационного риска. |

## P2 — улучшения

| # | Модуль | Замечание |
|---|---|---|
| 7 | [`wineid/canonical.py`](wineid/canonical.py:109) | [`canonical_query()`](wineid/canonical.py:102) вставляет выдуманный продюсер `«Неизвестный»`, противореча докстрингу «Unrecognized words are not invented». Для CLIP-эмбеддинга запроса это меняет семантику (строка «производитель неизвестный»). Предложение: подставлять пустой хвост, а не плейсхолдер. |
| 8 | [`wineid/search.py`](wineid/search.py:237) | `Candidate.fuzzy_rank` всегда равен `rank` — мёртвое поле, вводит в заблуждение при чтении вывода. |
| 9 | [`wineid/ocr.py`](wineid/ocr.py:20) | Поля `boxes` и `confidence` захардкожены `None` — мёртвые поля контракта; либо удалить, либо документировать как «не заполняется». |
| 10 | [`requirements.txt`](requirements.txt:5) | `python-Levenshtein>=0.25` в зависимостях, но нигде не импортируется (свой `levenshtein` в `search.py`); лишняя GPL-зависимость. |
| 11 | [`wineid/catalog_export.py`](wineid/catalog_export.py:19), [`wineid/clip_eval.py`](wineid/clip_eval.py:14), [`wineid/train_clip.py`](wineid/train_clip.py:19), [`tests/test_search_v2.py`](tests/test_search_v2.py:3) | Неиспользуемые импорты (pyflakes): `PIL.Image`, `numpy as np`, `hashlib`, локальные `json/time`, `load_catalog`. |
| 12 | [`wineid/canonical.py`](wineid/canonical.py:35) | `VOLUME_RE` не распознаёт «1000 мл» и другие 4+ значные объёмы — токен `1000` попадает в имя. Мелочь, но расходится с комментарием «Volume patterns such as…». |
| 13 | `tests/*` | **Пробелы покрытия HTTP/OCR:** нет тестов на 413 при превышении `MAX_BODY`, отсутствие/невалидный `Content-Length`, chunked body, 404 на незнакомых путях, `/ready` → 503, медленное тело (slow-body), OCR 5xx/4xx/invalid_response/no_text/deadline, `ocr_not_configured`, deadline между шагами `pipeline.predict`. Негативные кейсы `roi.decode` (RGBA/LA, DecompressionBomb) частично есть, но не полны. |
| 14 | [`wineid/visual.py`](wineid/visual.py:128) | `os.link` для атомарной публикации не работает на ФС без hardlink (FAT/exFAT/некоторые сетевые) — стоит добавить fallback `os.rename`. Аналогично в [`wineid/clip_zero_shot.py`](wineid/clip_zero_shot.py:158). |
| 15 | [`wineid/data.py`](wineid/data.py:85) | `audit()` и [`catalog_export.image_filters()`](wineid/catalog_export.py:46) читают файл целиком в память до проверки `MAX_BYTES` внутри `decode`. Для 12 MB+ файлов — ненужные пики памяти; проверять `stat().st_size` до `read_bytes()`. |
| 16 | [`wineid/server.py`](wineid/server.py:80) | Необработанные исключения вне `ValueError/UnicodeError/socket.timeout` (например, ошибка PIL при странном содержимом) → 500 с traceback в stderr. Желательно общий catch → `{"status":"error","reason_codes":["internal"]}` без деталей. |
| 17 | [`wineid/media.py`](wineid/media.py:49) | Ключ `derivative_groups` строится по `casefold()`-имени: регистрозависимые различия «гипотезы суффикса» схлопываются — вероятно, ок для инвентаризации, но стоит зафиксировать в комментарии. |
| 18 | [`wineid/search.py`](wineid/search.py:194) | Fuzzy-расширение пула — `O(N · snippets · docs)` на запрос при коротком пуле; для 2103 SKU допустимо, но на больший каталог стоит перейти на префиксный/токенный пред-фильтр. |

## Проверки и результаты

- `python3 -m pytest -q` → **65 passed in 1.91 s** (соответствует `IMPLEMENTATION.md`).
- `python3 -m compileall -q wineid tests` → OK.
- `pyflakes wineid tests` → находки перечислены в P2-11 (+ `producers` в P1-1).
- Реальный inference/OCR/обучение не запускались (нет весов/ключей/разметки) — по условиям проекта.

## Соответствие документации (сверка код ↔ документы)

| Документ | Статус |
|---|---|
| `IMPLEMENTATION.md` §ТЗ: «веса негативов: тот же производитель ×2» | **Расходится** — в torch-тренировке не реализовано (P1-1) |
| `IMPLEMENTATION.md`: числа 2103/1375/1699/404/41/216/64/3 | Консистентны с фильтрами [`catalog_export.py`](wineid/catalog_export.py:41) |
| `IMPLEMENTATION.md` §ТЗ: «проекция учится отдельно, AdamW» | Реализовано ([`train_clip.py`](wineid/train_clip.py:238)) |
| `IMPLEMENTATION.md`: «LoRA на Q/K/V+out-proj последних 4–6 блоков» | Реализовано, но маппинг на Jina не проверен (P1-5) |
| `plans/STATUS.md`: 65 тестов, запрет автоматического ROI, Jina не подключён к HTTP | Подтверждено кодом |
| `JINA_CLIP.md` / zero-shot контракты | Подтверждено: `ambiguous_prompt`, `ranked_only`, аудит до split, `confidence_calibrated=None` |
| `plans/STATUS.md` B06: «slow body/OCR, перегрузки» | Подтверждено как открытое (P1-4, P2-13) |

## Положительные практики (зафиксировать как эталон)

1. **Абстенция вместо угадывания:** `Pipeline`/`decide` возвращают `ambiguous`/`unknown` без калиброванной policy; singleton не принимается визуально ([`clip_eval.py`](wineid/clip_eval.py:114)).
2. **Аудит всего манифеста до split** с проверкой реальных SHA-256 файлов ([`data.py`](wineid/data.py:15), [`clip_class_eval.py`](wineid/clip_class_eval.py:35)) и отказ от загрузки весов до аудита ([`clip_zero_shot.py`](wineid/clip_zero_shot.py:376)).
3. **Provenance/checksum-дисциплина:** fingerprint policy, `archive_sha256`, `manifest_sha256`, атомарная публикация npz без перезаписи.
4. **Безопасность путей:** `is_relative_to`, запрет `..`, symlink-эскейпов; лимиты пикселей/байт в [`roi.py`](wineid/roi.py:8).
5. **Секреты не логируются:** `log_message` подавлен, ключ не входит в repr результатов ([`ocr.py`](wineid/ocr.py:153)).
6. Тесты с инъекцией эмбеддингов без скачивания весов (fixture-контракты HF проверяются моками).

## Рекомендуемый порядок закрытия

1. **До любого запуска обучения:** P1-1 (применить веса производителей в `clip_loss` или честно убрать из документации), P1-2 (контракт пути), P1-5 (smoke на реальной модели).
2. **До интеграции в HTTP:** P1-4 (auth/потоки), P1-6 (единый бэкенд поиска), P2-16 (catch-all 500).
3. **Параллельно:** P2-7…P2-12 (чистка), P2-13 (дописать негативные тесты HTTP/OCR/pipeline), P2-15 (лимиты до чтения).
4. **После решения B01–B04:** калибровка весов блендера [`search._blend`](wineid/search.py:241) и порогов policy на независимой validation — сейчас это заглушки.

---

*Отчёт является результатом статического ревью; не заменяет независимую оценку качества моделей, пентест и нагрузочное тестирование (см. gates в `plans/STATUS.md`).*