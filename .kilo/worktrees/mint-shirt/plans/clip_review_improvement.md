# CLIP: ревью реализации и план улучшений

## Актуализация после ревью

Этот документ ниже сохраняет **исторические findings**, а не текущий список дефектов. Единый актуальный backlog — [STATUS.md](STATUS.md). Повторная проверка: **35 passed**; качество на независимых фото по-прежнему не измерено.

Закрыто в текущем `clip_eval`: margin по сильнейшему визуальному конкуренту и отказ для singleton; versioned VisualPolicy с конфигурацией/calibration provenance; аудит полного query manifest до split; парные проверки и разделение общего/условного Recall@K. Rerank переведён с OpenCLIP на Jina. Эти P1 не нужно реализовывать повторно.

Открыто: общий аудит остальных веток ещё не унифицирован; text Policy не имеет аналогичной привязки; нет verified pilot gallery/независимого качества, полного environment lock, сервисного resource budget и реального CAM smoke-test. OpenCLIP-specific preprocessing замечания ниже относятся к прежнему rerank; текущую Jina-геометрию проверять отдельной абляцией.

## Историческая область и вывод

Проверены `wineid/visual.py`, `wineid/clip_eval.py`, `wineid/clip_roi.py`, связанные тесты, зависимости и планы 05/Grad-CAM. Это два независимых offline-направления: image-to-image rerank и эксперимент локализации. Их не следует одновременно внедрять в основной pipeline.

**Рекомендация:** сначала исправить достоверность эксперимента и принятия решений, затем измерить пользу frozen CLIP на проверенных данных. Смена backbone, fine-tuning и GrabCut сейчас не приоритет.

Проверка текущего состояния: `python3 -m pytest -q` → **22 passed**. Тесты CLIP используют фиктивные embeddings/CAM: они не проверяют реальный OpenCLIP forward/backward или качество распознавания. В рамках ревью веса не скачивались, реальные модельные эксперименты не запускались.

### Что уже хорошо

- Контракт вручную проверенной gallery и проверки provenance/hash, безопасных путей, ролей, размерностей и нормализации. Это не означает наличия реальной verified gallery в проекте.
- Batch-построение gallery и эксклюзивное создание архива без перезаписи.
- Missing reference не превращается в нулевой cosine; без policy нет visual acceptance.
- Парное сравнение с text-only, helped/harmed query, знаменатели метрик.
- Grad-CAM отделён от сервера; letterbox, возврат цветного crop, удаление hooks в `finally`, явные отказы.

## Findings

### P1 — неправильный конкурент при проверке visual margin

**Место:** `wineid/clip_eval.py:visual_decision`.

В fusion-режиме cosine победителя сравнивается только со вторым кандидатом **fusion**, а не с наиболее похожим визуально альтернативным кандидатом. Третий кандидат может иметь более высокий cosine, но решение будет `accepted`.

Воспроизведено: text scores A/B/C = 100/90/0, cosine = 0.8/0.5/0.95, weight = 0.5. Fusion возвращает A/B/C. Policy `(min_text=0, min_cosine=0.5, min_margin=0.2)` принимает A, несмотря на cosine C=0.95.

**Исправление:** явно определить семантику margin. Для visual verification сравнивать победителя с `max(cosine остальных Top-K)`; margin fusion, если нужен, сделать отдельным параметром. Для singleton Top-K определить отдельное правило — отсутствие конкурента не является доказательством отличимости. Добавить регрессию с ≥3 кандидатами.

### P1 — policy не привязана к эксперименту

**Место:** `wineid/clip_eval.py:VisualPolicy`, `evaluate`, `main`.

Policy содержит произвольную строку версии и три порога. Нет проверки её происхождения с validation или совместимости с gallery, моделью, preprocessing, ролью, K, mode и weight. В JSONL результата не сохраняются mode, weight, role и значения порогов. Одинаковая `version` может обозначать разные решения; policy можно без предупреждения перенести на другой эксперимент или подобрать на test.

**Исправление:** versioned policy artifact с полным конфигом и hash, fingerprint модели/gallery/preprocessing, hash calibration manifest и списком source groups. Проверять совместимость и непересечение calibration/test. Сохранять run metadata рядом с JSONL. Это делает нарушения обнаружимыми, но не заменяет организационное правило untouched test.

### P1 — неполный контроль утечек validation/test

**Место:** `wineid/clip_eval.py:evaluate`.

Query фильтруются по split до аудита. Проверяется пересечение выбранного split с gallery, но не повторение source group, query ID или изображения между validation и test. Дубликаты/ресайзы с разными SHA также не обнаруживаются автоматически. В `clip_roi.experiment` есть проверка source groups между splits, но нет общего аудита image/duplicate groups.

**Исправление:** единый валидатор всего manifest до фильтрации. Проверять ID, image hash, source/session/duplicate groups, gallery↔query и calibration↔test. Near-duplicate detection использовать для списка ручной проверки, не для автоматического подтверждения независимости.

### P2 — preprocessing недостаточно воспроизводим и может обрезать этикетку

**Место:** `wineid/visual.py:PREPROCESS`, `CLIPEmbedder`; `requirements-clip.txt`.

`open_clip-default` — не описание фактических transforms. Диапазон `open_clip_torch>=2.26,<4` допускает разные реализации. Стандартный center crop для выбранной модели может удалить края неквадратного ROI; letterbox из Grad-CAM не применяется к rerank. Это риск качества, а не измеренная деградация.

**Исправление:** сохранить фактические resize/crop/interpolation/mean/std, версии библиотек и checksum checkpoint. Сравнить current preprocessing с aspect-preserving padding на одинаковых query/gallery. Не заменять без ablation; каждое изменение требует новой gallery и policy. Фиксировать проверенное окружение lock/constraints-файлом.

### P2 — покрытие и метрики могут затруднить интерпретацию

**Место:** `wineid/clip_eval.py:rank`, `comparison`.

- Любой missing reference отключает rerank всего Top-K. Это безопасная текущая политика, но её влияние на coverage нужно выделять, а не скрывать.
- `gt_in_text_top_k` считается только среди query с доступным rerank: название выглядит как общий text Recall@K, хотя это условная подвыборка.
- `comparison` превращает строки в dict без проверки дубликатов и не проверяет согласованность GT/source между парой результатов при самостоятельном вызове.
- Нет доверительных интервалов, срезов по похожим урожаям/ролям/числу эталонов и отдельной доли ухудшений среди правильных text top-1.

**Исправление:** отдельно показывать общий text Recall@K, eligible Recall@K, причины missing coverage, decision accuracy и ranking accuracy. Добавить парную проверку входов, helped/harmed fractions и cluster bootstrap по source group. Text-only fallback исследовать как отдельную policy, сохраняя признак отсутствия визуального подтверждения.

### P2 — производительность ещё не измерена как у сервиса

**Место:** `wineid/visual.py:CLIPEmbedder`, `Gallery.scores`; `wineid/clip_eval.py:evaluate`.

Модель работает на CPU по умолчанию, query обрабатываются по одному. `scores` вычисляет cosine со всей gallery, хотя возвращает только Top-K. Это не обязательно bottleneck при текущем размере каталога: сначала профилирование. Deadline ручного ROI не ограничивает последующий inference. `timings_ms.total` — сумма offline стадий с повторной подготовкой ROI, не реальная end-to-end latency сервиса.

**Исправление:** явно задавать device/precision, измерять warm/cold latency и память; при GPU синхронизировать таймеры. Кэшировать query embeddings по hash crop + fingerprint encoder для перебора policy. Оптимизировать scoring Top-K только при измеренной необходимости. Перед HTTP-интеграцией определить общий budget и механизм ограничения занятого worker: проверка времени после forward не прерывает вычисление.

### P2 — Grad-CAM проверен только на геометрических фикстурах

**Место:** `wineid/clip_roi.py:CLIPCAM`, `experiment`; `tests/test_clip_roi.py`.

- Тесты не вызывают настоящий `CLIPCAM`: совместимость hooks, layouts и ненулевые patch gradients не подтверждена.
- Параметры модели остаются `requires_grad=True`; backward считает также ненужные для frozen CAM градиенты весов. Есть потенциал экономии памяти, требующий проверки эквивалентности CAM.
- Одновременные вызовы одного instance используют общую модель/hooks/zero_grad; instance нельзя считать thread-safe.
- `experiment` требует bbox у каждого изображения: полноценный тест сцен без этикетки текущим контрактом не описывается.
- В строках результата отсутствует полный конфиг checkpoint/prompts/threshold; часть сведений остаётся только в stdout.

**Исправление:** opt-in smoke-тест с одобренным локальным checkpoint, тест ненулевых градиентов, повторных вызовов и удаления hooks при исключении. Заморозить веса с сохранением input gradients и сравнить результат. До внедрения — сериализация вызовов или отдельные workers. Расширить GT явным наличием/отсутствием этикетки; считать false localizations на negatives. Сохранять полный run artifact. GrabCut добавлять только после доказательства полезности CAM.

## План выполнения

| Этап | Задачи и результат | Критерий завершения |
|---|---|---|
| 1. Корректность, P1 | Исправить margin, правила singleton; общий аудит manifests; схема policy/run metadata; тесты несовместимости | Сценарий A/B/C больше не принимается как visual verified; cross-split leakage отклоняется; изменение weight/role/gallery требует совместимой policy |
| 2. Воспроизводимость, P2 | Fingerprint preprocessing/checkpoint, фиксированное окружение, сохранение полного config; строгая валидация candidate scores и метаданных gallery | NaN/Inf, повреждённые metadata, duplicate assets и несовместимые artifacts дают понятные ошибки; run можно повторить из сохранённого конфига |
| 3. Данные и baseline | Verified pilot gallery; независимые query с known/unknown/ambiguous, похожими урожаями, бликами и разными ролями; аудит групп; baseline с реальным OCR | Есть отчёт о покрытии SKU и числе независимых групп; validation/test изолированы; ручной текст помечен как oracle |
| 4. Парные ablations | На одном наборе сравнить text-only, CLIP-rerank, fusion; отдельно current preprocessing vs padding; подобрать weight/пороги только на validation | Отчёт с accuracy, precision/coverage, unknown false accepts, harmed/helped, CI и latency; политика заморожена до test |
| 5. Решение о внедрении | Один прогон выбранной конфигурации на untouched test; затем при успехе — shadow mode и интеграция с budget | Польза относительно text-only при заранее согласованных ограничениях на false accepts, ухудшения и p95; иначе CLIP остаётся offline |
| 6. Отдельный CAM-пилот | Реальный smoke-тест, negatives, сравнение manual ROI / OpenCV / CAM на тех же фото; OCR и итоговое распознавание | Улучшается downstream-качество при допустимых отказах, latency и памяти; одного IoU недостаточно |

Этапы 1–2 можно выполнять без скачивания весов; подготовку данных вести параллельно. Этапы 4–6 требуют разрешённого checkpoint и реальной разметки. Численные production-пороги согласовать до тестирования, а не выбирать по получившемуся test результату.

### Минимальная тестовая матрица

- Margin: ≥3 кандидатов, visual/fusion disagreement, равные cosine, singleton, пустой Top-K.
- Policy: несовместимые mode/weight/role/K/gallery/preprocessing; отсутствие calibration provenance.
- Данные: группы и hashes между splits/gallery, duplicate query IDs, несовпадение GT в paired report.
- Embeddings: NaN/Inf/zero/wrong dimensions, повреждённый archive, дубликаты provenance.
- Геометрия: wide/tall ROI, EXIF, alpha, WEBP, сохранение текста на краях после transforms.
- CAM: настоящий backward opt-in, hook cleanup при ошибке, repeated calls, negatives, deadline overrun.

### Не делать до результатов

Не обучать CLIP, не вводить ANN или сложный калибратор, не объединять CAM и rerank на одном checkpoint ради экономии без измерений. Не считать cosine вероятностью или визуальное сходство доказательством точного урожая. Разное число reference на SKU может смещать max-cosine; сначала измерить этот эффект по срезам, затем сравнивать альтернативы агрегации.
