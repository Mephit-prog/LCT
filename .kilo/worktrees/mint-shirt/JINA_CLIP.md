# Jina CLIP v2: zero-shot без OCR

Реализовано в `wineid/jina_clip.py` и `wineid/clip_zero_shot.py`:

- **Русский запрос → фотографии**: независимо кодируем текст и фото, ранжируем по cosine.
- **Фото → классы каталога**: классы задаются текстовыми описаниями; эталонные фотографии не нужны. Для независимой разметки добавлен offline evaluator `evaluate-classes` (ранжирование, не policy принятия).
- Никакого обучения, OCR, fuzzy search или обязательного выделения ROI в этом пути нет.
- `wineid.visual.CLIPEmbedder` / `wineid.clip_eval` тоже переведены на Jina для прежнего image-to-image эксперимента. Это отдельная задача, не zero-shot классификация.
- HTTP/OCR pipeline не изменён. OpenCLIP в `wineid.clip_roi` оставлен **только для старого Grad-CAM эксперимента**.

## Установка и ограничения

```bash
python3 -m pip install -r requirements-clip.txt
mkdir -p artifacts
```

Модель: [`jinaai/jina-clip-v2`](https://huggingface.co/jinaai/jina-clip-v2), локальный `transformers.AutoModel`, нативные `encode_text` / `encode_image`. Вход модели 512×512, выход 1024 измерения. Сначала EXIF-ориентация и RGB (прозрачность на белом), затем штатный image processor модели. Он может менять геометрию/обрезать кадр; это **не детектор этикетки**. При необходимости используйте заранее подготовленные crops и сравните их с полным кадром отдельно.

Веса — **CC BY-NC 4.0**: для коммерческого использования требуется отдельное разрешение/лицензия Jina. `trust_remote_code=True` выполняет Python-код Hugging Face на вашей машине. Поэтому CLI требует `--allow-remote-code`, Python API — `allow_remote_code=True`. Флаг разрешает скачивание и исполнение кода, а не отменяет условия лицензии. Изображения передаются локально в PIL, а не внешнему API; URL изображений адаптер не принимает.

Весовой snapshot и основной HF code snapshot зафиксированы commit SHA в `wineid/jina_clip.py`; токенизатор и processor загружаются с теми же версиями, а не лениво с `main`. **Upstream дополнительно загружает config/code текстового backbone транзитивно; полного lock всех зависимостей здесь нет.** Для воспроизводимого изолированного запуска нужны проверка этих зависимостей, фиксация окружения и подготовленный HF cache. Один флаг не делает remote code безопасным.

По умолчанию используется CUDA при наличии, иначе CPU; `--device cpu|cuda|mps` задаёт устройство явно. FP32, без обязательных FlashAttention/xFormers; `--batch-size 8`, для ограниченной памяти уменьшите до 1–2. Модель ~0.9B параметров: это несколько GB памяти, CPU может быть медленным. MPS/CUDA и реальная производительность в этой реализации пока не проверены.

## 1. Поиск фотографий по русскому тексту

Положите свои фотографии в `wine_images/` (подкаталоги допустимы). Поддерживаются JPEG, PNG, WEBP; формат проверяется по содержимому. Файлы с повреждёнными данными не пропускаются молча. Имена файлов используются только как идентификаторы результата, **не как описание класса или разметка вина**.

```bash
# Индекс изображений строится один раз.
python3 -m wineid.clip_zero_shot build-images wine_images artifacts/images.npz \
  --allow-remote-code --batch-size 8

python3 -m wineid.clip_zero_shot search artifacts/images.npz \
  'Фотография бутылки вина «Анджуйское»; видна лицевая этикетка.' \
  --top-k 10 --allow-remote-code
```

JSON содержит `image_root`, `candidates[].id` (относительный путь), `image_sha256`, `rank`, `cosine`. Поиск не открывает фотографии повторно и не вычисляет их эмбеддинги заново. После изменения состава/содержимого фото пересоберите индекс в **новый** файл. Для проверки существующих фото по сохранённым SHA256 без загрузки модели используйте `python3 -m wineid.clip_zero_shot audit-images artifacts/images.npz`. Поиск сам не проверяет файлы на диске.

## 2. Zero-shot предсказание класса

```bash
# 2103 уникальных Slug текущего CSV; name = «Название вина».
python3 -m wineid.clip_zero_shot build-classes artifacts/classes.npz \
  --csv strapi_output0709.csv --prompt name --allow-remote-code

python3 -m wineid.clip_zero_shot predict artifacts/classes.npz \
  wine_images/bottle.jpg --top-k 5 --allow-remote-code
# Можно указать несколько файлов после пути к индексу.
```

`predicted_class_id` — закрытое Top-1 предсказание (при CSV это slug); оно **не означает подтверждённое распознавание**. При идентичных prompts разных классов `predicted_class_id: null`, статус `ambiguous_prompt`, а `indistinguishable_class_ids` содержит все неразличимые slug. В остальных случаях статус `ranked_only`; `confidence_calibrated: null` всегда. Модель выбирает среди предложенных классов даже для неизвестного вина; open-set rejection не реализован. Равные cosine разрешаются стабильным порядком индекса, а не считаются доказательством различимости классов. Softmax-проценты не выдаются.

### Одинаковые русские шаблоны

`--prompt`:

- `title`: точное название (контрольный эксперимент).
- `name` (по умолчанию): `Фотография бутылки вина «{name}»; видна лицевая этикетка.`
- `name-producer`: тот же шаблон с `производителя {producer}`.
- `name-producer-visual`: `Фотография бутылки вина «{name}» производителя {producer}: {visual}.`

`--ensemble` усредняет нормированные эмбеддинги двух однотипных русских формулировок на класс и повторно нормирует среднее. Для `title` формулировка одна. Улучшение от ensemble не гарантируется.

Нет автоматического перевода, транслитерации, добавления страны происхождения, цвета этикетки или инструкции `retrieval.query` (она добавила бы английский префикс). Названия сохраняются буквально, включая собственные имена на латинице из каталога; если нужны строго кириллические описания, подготовьте проверенный русскоязычный JSONL, не машинные догадки.

Производитель/визуальные признаки должны быть заполнены **у всех классов** выбранного шаблона, иначе сборка завершается ошибкой. Если производитель неизвестен хотя бы у одного кандидата, используйте `name` для всех. Признаки берутся только из вручную подготовленного JSONL, не из названия/slug:

```json
{"class_id":"wine-a","name":"Анджуйское","producer":"Проверенное имя производителя","visual":"проверенное описание лицевой этикетки"}
```

Это схема, а не реальные сведения о вине. Замените значения своими данными и добавьте по одной строке на каждый класс:

```bash
python3 -m wineid.clip_zero_shot build-classes artifacts/classes-visual.npz \
  --classes artifacts/classes.jsonl --prompt name-producer-visual \
  --ensemble --allow-remote-code
```

Индексы хранят сами prompts, hash CSV/JSONL, версии модели/processor и тип (`images`/`classes`). Индексы разных типов, моделей и версий не смешиваются; существующий файл не перезаписывается. Старые OpenCLIP gallery и policy необходимо пересобрать/перекалибровать.

## 3. Recall@1 / Recall@5 для text-to-image

Нужны независимые проверенные соответствия запросов фотографиям. Создайте `artifacts/retrieval-validation.jsonl`:

```json
{"query_id":"q1","text":"Фотография бутылки вина «Точное название»; видна лицевая этикетка.","relevant_image_ids":["bottle.jpg"],"hard_negative":true}
```

`relevant_image_ids` — пути относительно корня индексируемой папки, размеченные человеком; один запрос может иметь несколько релевантных фото. Отметка `hard_negative` означает, что в индексе есть проверенные похожие конкуренты; код сам их не определяет.

```bash
python3 -m wineid.clip_zero_shot evaluate artifacts/images.npz \
  artifacts/retrieval-validation.jsonl --allow-remote-code \
  > artifacts/retrieval-report.json
```

Считается hit-style Recall@K: доля запросов, у которых **хотя бы одно** релевантное фото находится в Top-K. Есть числители/знаменатели для всех запросов и hard-negative subset. Это не precision, не доля найденных релевантных документов и не accuracy классификации фото. Вино без релевантных фото этим отчётом не оценивается.

Сравнивайте шаблоны на одном validation-наборе, отдельно похожие названия/этикетки и вина одного производителя. После выбора шаблона, модели и состава кандидатов зафиксируйте их и оцените на отдельном test. Split/source-group audit для этого простого JSONL не автоматизирован: независимость и отсутствие дублей должен обеспечить автор набора. `eval/` проекта — только проверка интерфейса, не источник GT/настройки.

## 4. Offline photo→SKU evaluation (размеченные фото)

CSV уже задаёт ground truth для **каталоговых записей** (`Slug`, `Название фото`), но не размечает независимо снятые query. Чтобы воспроизводимо искать локальные original assets, не выдавая сходство имён за подтверждённую media relation, выполните:

```bash
python3 -m wineid.catalog_media strapi_output0709.csv \
  prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads \
  artifacts/csv-media-links.jsonl > artifacts/csv-media-links-summary.json
```

Артефакт содержит CSV slug/номера строк/SHA256 и кандидатов по имени фото и/или slug, а также 404 unresolved на текущем наборе. У совпадений `mapping_status=candidate`, **не** `verified`; из этого файла нельзя автоматически делать validation/test manifest. Требуется подтверждение пары байтов↔SKU и новые снимки query.

Составьте отдельный JSONL рядом с независимыми фото. `image_path` — путь **относительно JSONL**, `image_sha256` — SHA256 реальных байтов фото, а не index; `source_group` объединяет съёмку и производные кадры. GT задаётся человеком: `known` только при проверенном slug из class index; у `unknown` и `ambiguous` `gt_slug: null`. Поле `gt_source` обязательно для **всех** статусов. Например (замените хеш, ID и путь своими данными):

```json
{"query_id":"shoot-1-photo-1","image_path":"photos/bottle.png","image_sha256":"<64 hex chars>","source_group":"shoot-1","split":"validation","gt_status":"known","gt_slug":"verified-catalog-slug","gt_source":"independent human annotation"}
```

```bash
python3 -m wineid.clip_zero_shot evaluate-classes artifacts/classes.npz \
  artifacts/sku-validation.jsonl --split validation --allow-remote-code \
  > artifacts/sku-name-report.json
# После сборки второго class index из ТОГО ЖЕ snapshot каталога и оценки тех же фото:
python3 -m wineid.clip_zero_shot compare-classes \
  artifacts/sku-name-report.json artifacts/sku-name-producer-report.json
```

До загрузки модели проверяются **все** строки manifest (включая другие split): уникальные ID и байты фото, фактический SHA256, GT и членство known slug в class index, группы и дубли между split, безопасные относительные пути. Неразрешённые случаи нельзя выдавать за unknown. Фото для выбранного split декодируются как ориентированный EXIF full frame, без ROI. Index prompt/crop и набор кандидатов сохраняются в отчёте вместе с хешами manifest, index и исходного CSV/JSONL. `compare-classes` работает без модели, но требует одинаковые query manifest, class IDs, SHA256 источника классов, encoder и способ обработки фото; выдаёт `helped/harmed` по точному Top-1.

Отчёт разделяет known/unknown/ambiguous: Top-1/5 **только среди known**, отдельно доля известных классов с идентичными/нормализованными prompts, ошибочный Top-1 вне prompt-группы верного класса и raw Top-1 при коллизиях. Raw ранги при одинаковых prompts разрешены порядком индекса; это **не** точное подтверждение SKU. Интервалы Top-1/5 — диагностический bootstrap по независимым `source_group` (отсутствуют при <2 группах), не гарантия. Для unknown/ambiguous пока нет калиброванного отказа: `false accept`, precision принятия и качество сервиса **не оцениваются**. При отсутствии собственного размеченного пилота запуск не даёт метрик качества; не используйте `eval/` и связи по имени uploads как GT. Сравнение native/pad/ROI потребует раздельно проверенных preprocessing/index и парного протокола, а не простой подмены изображений.

## Python API

```python
from wineid.jina_clip import JinaCLIPEmbedder
from wineid.clip_zero_shot import EmbeddingIndex

encoder = JinaCLIPEmbedder(allow_remote_code=True, device="cpu", batch_size=2)
index = EmbeddingIndex.load("artifacts/images.npz")
for result in index.search('Этикетка вина «Анджуйское»', encoder, top_k=5):
    print(result["cosine"], result["id"])
```

## Проверено

`python3 -m pytest -q`: **45 passed**. Тесты используют фиктивные эмбеддинги/модель: проверяют batching, cosine, prompt ensemble, HF API arguments, EXIF/RGB, сохранение индексов, несовместимости и Recall. Дополнительно выполнен реальный CPU inference на трёх файлах `eval/queries` с весами revision `e10d47f...` и `transformers==4.49.0` в отдельном окружении; результаты — в `artifacts/eval-jina-predictions.json`, индекс фото — в `artifacts/eval-jina-images.npz`. Это проверка выполнения, **не оценка качества**: у `eval/` нет доступных правильных ответов. В текущем CSV 444 из 2103 классов попадают в 100 групп с одинаковым `{name}`; их невозможно различить с шаблоном `name`. Мультиязычность запроса **не гарантирует чтение мелкой кириллицы на фото**; похожие этикетки остаются обязательным отдельным тестом.
