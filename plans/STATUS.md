# Текущий статус и единый backlog

### Финализация — checkpoint после работы по [FINALIZATION.md](FINALIZATION.md)

**Новый checkpoint: продолжена локальная часть PR-04 (изолированные CLIP и warm Paddle worker); RC и независимый gate качества НЕ пройдены. Приёмка PR-04/05/06/07/08 остаётся открытой.** Облачный MinerU развивается отдельно и не подключён к HTTP wineid.

- **PR-01 (локально проверен):** устранены конфликты Pillow/OpenCV в `constraints.txt`, добавлены `requirements-dev.txt`, Python 3.12 offline job `.github/workflows/offline.yml`, исправлены pyflakes и deprecated `Image.fromarray(mode=...)`. Scoring fuzzywuzzy закреплён за `difflib.SequenceMatcher` без `python-Levenshtein`. Чистый `.venv`: `pip check`, `compileall`, `pyflakes`, `pytest -q -rs` — **215 passed, 10 skipped, 2 warnings** (6 opt-in smoke, 4 torch-only без torch; оба warning от сторонних библиотек). Hosted CI ещё не запускался; ни реальные веса, ни сеть OCR не использовались.
- **PR-02 (локально реализован, target-проверки открыты):** `wineid/runtime.py` унифицирует bootstrap обоих HTTP-адаптеров, валидирует env/пути до загрузки модели, обеспечивает shutdown OCR даже при ошибке startup. Policy v3 связывает aliases, ROI, OCR, зависимости/код, индексы и weights, blend/conditional OCR, timeout и budget; несовместимая/legacy policy отклоняется при HTTP startup, overrides bbox/ROI/manual text не принимаются; мутации профиля и artifact тестируются. Нет проверки полного release bundle, canary на целевом сервере и покрытия всех вариантов частичного сбоя тяжёлых моделей.
- **PR-03 (кодовый parity-путь расширен, приёмка открыта):** `fusion_run` для не-oracle запросов использует `Pipeline.predict` с теми же ROI/OCR/vision и сохраняет сырые evidence, online status/reasons; тесты покрывают OCR ok/no_text/error, bbox, timeout, capacity, HTTP и ранжирование. `fusion_eval --policy` теперь повторно прогоняет **каждый** checked manifest по байтам с runtime, отвергает подмену evidence и ручной bbox/oracle; одиночные SHA внутри JSONL не аутентификация. Старые внешние вызовы `evaluate(..., policy=...)` принимают доверенные данные от вызывающего кода, не являются независимым доказательством инференции. Production parity с pinned весами/реальными ROI и просадкой качества не доказана.
- **PR-04 (часть, локальный mock IPC):** HTTP и `fusion_run` используют прогретый killable CLIP subprocess (ограниченный pipe, один запрос, startup canary, kill process group при timeout/crash, фоновый bounded restart), parent сохраняет class/gallery scores. Warm Paddle теперь хеширует веса и проходит canary только на startup/в фоновом bounded recovery с cooldown, а не на пользовательском запросе; при смене manifest/весов остаётся not-ready до нового согласованного запуска. Восстановление не блокирует HTTP; отказ worker даёт `503 ocr_unavailable`, forced eval не подменяет отказ каталоговым slug. Cold local OCR передаёт готовый PNG без повторной синхронной перекодировки. Policy fusion v4 различает isolated/inprocess, на потере vision не принимает text-only. Тесты mock IPC проверяют hang/crash/recovery/HTTP. **Не закрыто:** синхронные ROI/PNG-операции и скоринг в parent, полная ограниченность startup/shutdown при медленном хранении весов, Mistral общий timeout, совокупный CPU/GPU бюджет и wall-clock на целевой машине.
- **PR-05 (часть):** FastAPI получает Bearer и слот до чтения POST, ограничивает JSON/файлы и время chunked upload; stdlib не вычитывает тело при 401. Добавлены `/api/health/recognition`, `acceptance_ready` и typed visual-capacity busy. Не проверены reverse proxy, предел числа TCP соединений, многопроцессный лимит и реальная нагрузка.
- **PR-06/08 (заготовка):** `wineid.release_bundle` создаёт/проверяет локальный inventory SHA-256 файлов с запретом symlink/path traversal, `WINE_RELEASE_MANIFEST` запускает preflight до загрузки моделей, API/FRONTEND описывают новые статусы и ограничения. Inventory **не подписан**, не является полным lock, не закрывает HF/torch/OS/OCR веса, лицензии, clean checkout, target smoke/rollback. Документация и RC не заморожены.
- **Локальная проверка этой итерации:** Python 3.12 `.venv`: `pytest -q -rs` — 251 passed, 10 skipped, 2 внешних warnings; `pyflakes`, `compileall`, `pip check` — успешно. Hosted CI, real OCR/CLIP, внешнее API MinerU и target load **не** выполнялись.
- **Дальше:** PR-04 — ограничить parent preprocessing/скоринг, внешний OCR и подтвердить wall-clock deadline на target; PR-05 — инфраструктурный admission/нагрузка; PR-06/07/08 — выбранный target bundle, лицензии, реальные веса, smoke, нагрузки, runbook и проверенный rollback. До независимого GT/validation/test и product gates не объявлять MVP/RC или точный accepted slug. Существующие незакоммиченные файлы и артефакты, включая новую ветку MinerU, не коммичены и не удалялись.


Этот документ сохраняет **исторический** backlog инфраструктуры (ниже — результаты старых итераций). Актуальный путь online-бленда — [candidate_source_blend.md](candidate_source_blend.md), инструкция запуска — [API.md](../API.md). Реальный CLIP inference с LoRA и базовой Jina в новой итерации запущен на трёх неразмеченных фото `eval/` и 40 uploads с **кандидатной** связью CSV: [REAL_RUN.md](REAL_RUN.md). Платный OCR не запускался; галерея не подтверждена вручную, независимой оценки качества по настоящим фото нет.

### Отдельный документный трек: MinerU Precision cloud (opt-in, вне wine RC)

Добавлены `wineid/mineru_cloud.py`, `wineid/mineru_jobs.py`, `wineid/mineru_eval.py`, `requirements-mineru.txt`, [инструкция](../MINERU_CLOUD.md) и offline-контрактные тесты. CLI обрабатывает PDF/сканы отдельно от `/api/recognize`: сплит по страницам/байтам, ограниченная SQLite-очередь и локальный кэш, signed callback с проверкой, poll-recovery, безопасная выгрузка ZIP, RU-нормализация и отчёт CER/WER по ручному GT. Сеть включается только при `work --allow-upload` и заданном `MINERU_TOKEN`; тесты внешнего OCR не делают. Локально (Python 3.12 venv) `pip check`, `compileall`, `pyflakes`, `pytest -q -rs` — **225 passed, 10 skipped, 2 warnings**, десять новых тестов MinerU с MockTransport/файлами; CLI enqueue/status проверен без отправки. **Точка остановки по этому треку:** код и mock-контракты готовы к первому opt-in canary, но реальный MinerU токен/отправка/latency и точность `cyrillic`/`vlm` ещё не проверены; DPI-нормализация, объединение chunks, мониторинг и эксплуатации не закрыты. Историческая точка остановки wine-плана выше (PR-03) сохраняется; это не переоценка wine RC.

### Локальный OCR и сквозная текстовая ветка (эта итерация)

Добавлены экспериментальный прогретый Paddle worker через private IPC (`wineid/warm_ocr.py`), manifest локальных весов и версий (`wineid/ocr_manifest.py`), bounded I/O/timeout/kill+restart, canary/readiness и явная конфигурация provider без неявной отправки ROI во внешнюю сеть. FastAPI/stdlib HTTP используют тот же ROI → OCR → поиск/бленд с общим оставшимся deadline, `503 busy` без другой ветки. Доступна независимая диагностическая оценка ручных ROI с CER/WER и text Recall@5 (`wineid/ocr_eval.py`). Контракт проверен offline mock/IPC тестами; реальный Paddle с весами на целевом сервере, лицензии, независимый GT и нагрузка API+CLIP+OCR **не подтверждены**. Нет гарантии жёсткого wall-clock deadline для произвольного CLIP inference или общего лимита между несколькими API-процессами. Для включения см. [API.md](../API.md).

### Интеграция бленда (новая итерация)

Доступны явное подключение Jina class index + LoRA/verified gallery, union источников, RRF/global-gap, OCR-признаки с чисткой гомоглифов, группировка по сериям, forced top-1 для eval отдельно от основного API, экспериментальные TTA/галерея из синтетических видов, offline evaluation и versioned acceptance policy. Добавлен воспроизводимый offline `fusion_run`/`fusion_compare` на настоящих байтах с явной маркировкой unlabelled/candidate links; исправлен eval для большого валидного изображения, когда OCR crop слишком велик, но есть визуальные кандидаты. **По умолчанию модель не загружается и основной API не принимает slug без policy.** Отсутствие независимой validation/test/verified gallery и неизмеренный p95/SLA по-прежнему блокируют выпуск. Пункты и числа ниже — исторические snapshots и не доказывают результат этой итерации.

### Итерация инфраструктуры и backend (2026-09-26)

Выполнены исполнимые без данных пункты backlog плюс запрошенный фронтендером FastAPI-слой (в локальном рабочем окружении `python3 -m pytest -q`: **162 passed, 6 skipped, 0 failed**):

- **FastAPI backend** (`wineid/api.py`, `API.md`, `FRONTEND.md`, `requirements-api.txt`, `tests/test_api.py`): `GET /api/health`, `/api/wines[/{slug}]`, `/api/media/{filename}`, `/api/catalog/collisions`, `/api/aliases`, `POST /api/search`, `/api/recognize`, legacy `/ready`, `/v1/recognize`, `/v1/eval/predict`; CORS, Bearer-auth, ограниченные workers (`503 busy`), OpenAPI для фронта. Семантика распознавания не менялась.
- **B03** — общий аудит manifests: `data.retrieval_manifest_errors` (text→image), `data.calibration_test_leakage_errors`, подключены в `clip_zero_shot.retrieval_report` и `clip_eval.evaluate`.
- **B05** — `wineid/text_policy.py`: `TextPolicy` версионирована, с fingerprint/checksum, привязана к `csv_sha256`/`search_backend`/`blend_version`, хранит calibration hash/source groups и artifact checksum; `Pipeline` отклоняет несовместимую конфигурацию; `load_policy` читает новый или legacy артефакт.
- **B06** — `/ready` (stdlib) и `/api/health` разделяют `catalog_ready` и `recognition_ready`; ограничение конкурентных запросов и тест `503 busy` покрыты; устранена проблема TCP RST при раннем отказе в stdlib HTTP на Windows.
- **B07** — `constraints.txt` (пины offline-стека), `LICENSES.md` (Jina CC BY-NC 4.0, GPL-2.0, права), opt-in `tests/test_smoke_offline.py` (`WINE_SMOKE_OFFLINE=1`).
- **Фильтры данных и геометрия**: pHash-дедупликация (§4.1 ТЗ) в `wineid/phash.py` + `catalog_export`; ручной review near-duplicate/band-пар — `wineid/duplicates.py`; метрики сохранности ROI/полей (`label_retention`, нативный FOV Jina) — `wineid/roi_stats.py`; центральный 80% кроп (`center_crop_80_roi`, ТЗ §4.2); полная таблица алиасов производителей `producer_aliases.txt` (466 вариантов для всех 135 виноделен каталога) с автозагрузкой в `Index`, `server.py`, `api.py`.

Осталось из B06/B07: жёсткая гарантия wall-clock deadline для OCR и lock torch/transformers; B01/B02/B04/B08–B10 по-прежнему требуют решений владельца и независимого GT.

## Вердикт

**Работающий исследовательский прототип; пользовательский MVP и production не готовы.** Главный блокер — отсутствие независимой оценки сквозного качества. Процент готовности не является приёмочным критерием: готовность определяется gates ниже.

| Компонент | Реализовано / подтверждено | Не закрыто |
|---|---|---|
| Каталог и поиск | CSV-дедупликация, raw/provenance, fuzzy + Levenshtein; `catalog_media` связывает утверждённые в CSV имена фото с uploads как кандидатов | Независимый Recall@K и latency; подтверждённая media relation для локальных файлов |
| Jina zero-shot | CLI текст→фото и фото→классы, photo→SKU ranking evaluator с проверкой независимого manifest и парным сравнением, аудит prompt collisions; сохранены артефакты реального CPU inference | Unknown rejection/accept-policy, независимые размеченные фото и качество; не подключён к HTTP |
| Image-to-image | Jina rerank/fusion, visual policy fingerprints, полный аудит query manifest до split | Verified pilot gallery, измеренный выигрыш и калибровка |
| ROI | Ручной crop, экспериментальный OpenCV bbox; отдельный OpenCLIP CAM | Независимая оценка сохранности текста; реальный CAM smoke-test |
| OCR | Mistral adapter, ошибки/deadline checks, mock-тесты | Разрешённый реальный canary, качество, latency и стоимость |
| HTTP | Контракт, лимиты тела/пикселей, ограничение конкурентных OCR; FastAPI-адаптер (`wineid/api.py`) с CORS/bearer/OpenAPI, разделение catalog/recognition readiness | Жёсткий wall-clock budget, нагрузочная проверка, аутентификация/соединения для production |
| Данные и решение | CSV содержит каталоговый GT (`Slug`, имя/производитель, `Название фото`); контракты provenance/splits и механизмы отказа | Независимый GT для пользовательских query/unknown, подтверждённые связи локальных uploads с CSV, validation/test, согласованные пороги |
| Выпуск | Pinned Jina weights и основной HF code | Полный lock окружения/транзитивного кода, лицензии, эксплуатационная защита |

## Реализация по новым ТЗ (tz-clip-text-ranking.md, tz-wine-label-retrieval.md)

Начата кодовая реализация двух новых ТЗ (подробности и команды — в `IMPLEMENTATION.md`). Сделано без изменения контрактов: `python3 -m pytest -q` — **65 passed**.

- Канонический текст/нормализация (`wineid/canonical.py`), структурированные поля SKU (`wineid/catalog.py:structured_fields`).
- Переработанное текстовое ранжирование (`wineid/search.py`): IDF-токенное доказательство + лексический блендер по ТЗ §4.1; коллизии «одинаковые названия» разрываются сортом/сладостью при их наличии в тексте; неразрешимые группы возвращаются как `ambiguous` без принудительного Top-1 (по текущему CSV 55 групп / 116 SKU, 5,5%).
- Генерация `wines.csv`/`images.csv` (`wineid/catalog_export.py`): 2103 wine-строки (split по производителю 80/10/10), 1375 image-строк после фильтров ТЗ §4.1 из 1699 кандидатных связей (число без pHash-фильтра, повторный прогон нужен). pHash-дедупликация реализована (`wineid/phash.py`, Hamming < 6 между разными `wine_id` снимает обе связи); multi-bottle/watermark фильтры не реализованы (нужна внешняя разметка). Ручной review near-duplicate пар манифестов — `wineid/duplicates.py`.
- Vision-LoRA fine-tune с замороженной текстовой башней (`wineid/train_clip.py`) и canonical prompt-вариант zero-shot (`wineid/clip_zero_shot.py --prompt canonical`).

Открыто (требует данных/решений, не кода): verified gallery и независимая разметка, содержимое alias-таблицы производителей (механизм подключён), калибровка весов блендера/порогов на validation, лицензии и запуск LoRA на GPU.

## Границы известных результатов

- 45 тестов подтверждают проверяемые программные контракты, но не точность модели; CLIP-тесты используют заглушки.
- `eval/` — три фото без доступного GT, только интерфейс/выполнение. Не использовать для настройки.
- Три случая uploads и padding-абляция — диагностика по кандидатным связям локальных файлов с CSV, не validation. CSV принят как GT **каталога и поля `Название фото`**; локальные uploads переименованы, и одних нормализованных имён недостаточно для доказательства byte-level media relation. Внешний аудит в `jina_clip_failure_analysis.md` ошибочно сообщает об отсутствии CSV/артефактов: здесь они есть и хеши трёх фото совпадают; GT для независимых query и отдельный test отсутствуют.
- В `name`-prompt 444/2103 классов входят в 100 групп одинаковых prompts; в `name-producer` — 141/2103 в 64 группах по текущему CSV. Среди 1699 кандидатно связанных SKU это 389 и 118 соответственно (**не** частоты ошибок и не независимые query). `ambiguous_prompt` защищает только при попадании такой группы на Top-1; остальные ответы `ranked_only` не являются acceptance.
- Jina zero-shot, OCR HTTP, image-to-image и CAM — отдельные ветки. Нельзя описывать их как уже интегрированный сервис.

## Решения до следующего эксперимента

Владелец продукта совместно с ML/data ответственными должен зафиксировать:

1. Основной сценарий: поиск фото по тексту, показ кандидатов по фото или подтверждение точного SKU. Метрики этих задач не взаимозаменяемы.
2. Единицу ответа: точный `slug`, линейка или набор кандидатов; поведение для unknown, неоднозначного года и нескольких бутылок. Eval endpoint по-прежнему возвращает только точный slug либо null.
3. Допустимые wrong-SKU, accepted precision, coverage, unknown false accepts, p95 и стоимость; размер независимого test и способ расчёта интервалов.
4. Источник независимых фото/GT, ответственных за разметку и adjudication, целевое оборудование, право использования модели и передачи ROI OCR-провайдеру.

**Ответственные по ролям ниже — предлагаемые, конкретные люди и сроки ещё не назначены.** Не считать эти решения согласованными автоматически.

## Приоритетный backlog

| ID | Приоритет / ответственная роль | Задача | Приёмка |
|---|---|---|---|
| B01 | P0 / продукт + ML | Зафиксировать продуктовый контракт и метрики | Письменно определены сценарий, ответ/отказ, цели качества и бюджет; назначены владельцы данных и выпуска |
| B02 | P0 / data | Связать CSV-эталон с media; собрать независимый пилот known/unknown/ambiguous, похожих SKU и сцен с несколькими бутылками | Из CSV выгружено 2103 SKU и 1699 **кандидатных** media links (`artifacts/csv-media-links.jsonl`), 404 unresolved; 1350 кандидатов совпали по имени фото и slug, 289 только по имени фото, 60 только по slug. Подтвердить relation/ручной разметкой, не повышать candidate до verified автоматически; для query нужен свой GT с provenance, split по съёмкам/duplicate families до настройки, ручные ROI/транскрипции на части |
| B03 | P1 / backend + ML | Единый аудит manifests для OCR, visual и zero-shot | **Сделано:** общий `data.query_manifest_errors` (IDs, группы, GT, SHA между validation/test) для OCR, visual и photo→SKU; `data.retrieval_manifest_errors` подключён в text→image `retrieval_report`; `data.calibration_test_leakage_errors` используется и в visual evaluator, и напрямую. Осталось: применить к независимому GT (B02) |
| B04 | P1 / ML | Photo→SKU evaluator и аудит различимости каталога | Инструмент готов: Top-1/5 known с знаменателями и диагностическими source-group bootstrap интервалами; unknown/ambiguous отдельно, exact/normalized prompt collisions, paired helped/harmed и срез Recall@1 на pHash-парах 8–16 (ТЗ §8). Осталось применить к независимым GT, сравнить name/name-producer на одном validation и разобрать ошибочные SKU; до accept-policy это только ranking, не false-accept готового решения |
| B05 | P1 / backend + ML | Привязать текстовую policy к конфигурации | **Сделано:** `wineid/text_policy.py:TextPolicy` (versioned, fingerprint/checksum, `csv_sha256`+`search_backend`+`blend_version`, calibration hash/source groups); `Pipeline` отклоняет несовместимость; `check_calibration_test_leakage` отклоняет leakage; `load_policy` поддерживает legacy `Policy`. Осталось: калибровка порогов на реальной validation (B02/B08) |
| B06 | P1 / backend | Разделить readiness и обеспечить общий resource budget | **Частично сделано:** `catalog_ready` отдельно от `recognition_ready` в `/ready` и `/api/health`; ограниченные workers (stdlib semaphore, FastAPI `anyio.Semaphore`) с `503 busy`; негативные тесты тела/OCR/pipeline. Осталось: жёсткая гарантия wall-clock deadline (проверка после вызова не прерывает OCR) и нагрузочная проверка соединений |
| B07 | P1 / ML + инфраструктура + юридическая роль | Воспроизводимое разрешённое окружение | **Частично сделано:** `constraints.txt` пины offline-стека; `LICENSES.md` с лицензиями и открытыми решениями; opt-in offline smoke-test `tests/test_smoke_offline.py`. Осталось: lock torch/transformers после offline-прогона на целевой машине и решения по Jina CC BY-NC 4.0 / remote code / GPL-2.0 |
| B08 | P1 / ML | Минимальные paired ablations на одном validation | Native vs bounded white pad vs manual ROI; при необходимости проверить `longest+black` отдельно от white pad (не одно и то же); name vs name-producer при фиксированном crop; ручной текст vs разрешённый реальный OCR. Менять одну ось за раз; сохранять качество, отказы, время/память и версии |
| B09 | P2 / ML + backend | Выбрать и заморозить один путь, затем проверить untouched test | Полный независимый отчёт; acceptance/rejection policy откалибрована только на validation; при провале целей путь остаётся offline |
| B10 | P2 / backend + инфраструктура | Интеграция выбранного пути и ограниченный пилот | HTTP использует проверенную конфигурацию, измерен end-to-end p95 с очередью; shadow/pilot, наблюдаемость без утечки фото/OCR/секретов, согласованы условия доступа и rollback |

B03–B07 можно выполнять параллельно подготовке данных. Числовые gates из внешнего аудита `jina_clip_failure_analysis.md` — предложения, а не решение B01. B08 требует B01–B04 и разрешённого окружения/провайдеров. B09–B10 не начинать как выпуск без результатов предыдущих gates.

## Уже закрытые findings старых ревью

- Visual margin сравнивает победителя с максимальным cosine остальных кандидатов; singleton не принимается как визуально подтверждённый.
- VisualPolicy привязана к конфигурации/gallery и calibration provenance; сохраняется run metadata.
- `clip_eval` аудирует весь query manifest до фильтрации split; парные отчёты проверяют соответствия и разделяют общее/условное покрытие.
- Zero-shot диагностирует одинаковые prompts; CLI не выдаёт единственный slug при неоднозначном Top-1. Offline photo→SKU evaluator считает raw ranking отдельно от отказов и аудирует известные GT-коллизии.
- Добавлены `audit-images`, более строгая валидация индекса и batching retrieval evaluation; revision mismatch проверяется до загрузки модели.
- Реальный Jina CPU inference представлен сохранёнными артефактами. Это закрывает первоначальное отсутствие свидетельства выполнения, но не независимую оценку качества или полный lock окружения.

## Не включать в ближайший критический путь

CAM/GrabCut, fine-tuning, ANN, массовый внешний ETL и признаки вкуса. Image-to-image оценивать только при verified pilot gallery; её отсутствие не блокирует zero-shot и OCR baseline. Внешнее обогащение сначала ограничить проверенными различающими полями небольшого SKU-пилота, только если сравнение доступных каталоговых полей показало недостаточность.

## Gates готовности

- **Исследовательский baseline:** проверенные данные, воспроизводимый offline запуск и отчёт с полными знаменателями. Сейчас инструменты есть, независимый отчёт отсутствует.
- **Пользовательский MVP:** выбранный путь, согласованное качество на untouched test, корректные отказы, разрешённые лицензии и измеренная сквозная latency; HTTP реально использует этот путь.
- **Production:** дополнительно нагрузка/ресурсные лимиты, эксплуатационная защита, мониторинг, воспроизводимый выпуск и rollback. Прохождения mock-тестов для этого недостаточно.
