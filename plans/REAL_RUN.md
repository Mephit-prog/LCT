# Реальный локальный прогон (без подмены GT)

Проверено в этой рабочей копии: **Jina CLIP v2 (закреплённые веса из локального HF-кэша) + сохранённый LoRA** действительно исполняются на CPU, class index содержит 2103 SKU. `wineid.fusion_run` сохраняет сырые observations каждого источника, Top-20 и происхождение; `wineid.fusion_compare` сравнивает парные прогоны без превращения кандидатных ссылок в GT. Обе команды работают **без OCR-вызовов**, пока не задан явный `--ocr live --allow-live-ocr` и ключ. При `--ocr manual` текст из манифеста обозначается `manual_oracle`. Галерея допускается только по вручную verified manifest; из `images.csv` она автоматически не строится.

**Примечание к воспроизведению:** прогоны ниже выполнены до включения изолированного CLIP worker в `fusion_run` и его canary. После обновления кодовый fingerprint/профиль и версия fusion policy изменились; сохранённые отчёты остаются исторической диагностикой, а не evidence для acceptance. Для нового прогона нужен локальный pinned HF cache (worker запускается offline), новая директория вывода и отдельный замер затрат IPC.

## Что доступно и чего нет

- `strapi_output0709.csv` — настоящий каталог 2103 SKU; Strapi uploads находятся в `/Users/eu/Downloads/Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads`. `artifacts/tz/images.csv` — **эвристические кандидатные связи** с uploads, не вручную верифицированная разметка запросов; совпадения нельзя называть точностью.
- `eval/queries/*` — 3 **настоящие фотографии, но без GT**. Нельзя вычислить accuracy или калибровать acceptance на них. `artifacts/lora/jina-clip-v2-lora-synthetic.pt` (`sha256: 251ed86e97882ab328e8c41c25f0ddc0347f2cd2475280a01dcb1ee27443a25c`) существует; full-vision checkpoint отсутствует. База Jina cached, ревизии закреплены в `wineid/jina_clip.py`. Лицензия CC BY-NC 4.0, HF remote code и внешняя отправка ROI требуют разрешения перед коммерческим использованием.
- Нет независимого вручную размеченного набора реальных полевых фото (known/unknown/ambiguous), verified gallery и `MISTRAL_API_KEY` в текущем окружении. Поэтому платный canary, калибровка unknown и production policy **не запускались**.

## Воспроизведение без OCR

```bash
cd LCT
# Сначала разрешить лицензии и аудит HF remote code; --allow-remote-code обязателен.
python3 -m wineid.clip_zero_shot build-classes artifacts/classes-frozen-canonical.npz \
  --csv strapi_output0709.csv --prompt canonical --device cpu --allow-remote-code
python3 -m wineid.clip_zero_shot build-classes artifacts/classes-lora-canonical.npz \
  --csv strapi_output0709.csv --prompt canonical \
  --adapter artifacts/lora/jina-clip-v2-lora-synthetic.pt --device cpu --allow-remote-code
python3 -m wineid.fusion_run --images eval/queries/019c68d0.jpg \
  eval/queries/096ca74e.jpg eval/queries/02eef911.webp \
  --classes artifacts/classes-lora-canonical.npz \
  --adapter artifacts/lora/jina-clip-v2-lora-synthetic.pt \
  --allow-remote-code --device cpu --output artifacts/real-unlabelled-fusion-lora
# Реальные студийные фотографии с КАНДИДАТНОЙ связью к CSV:
UPLOADS=/Users/eu/Downloads/Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads
python3 -m wineid.fusion_run --candidate-images artifacts/tz/images.csv --uploads "$UPLOADS" \
  --sample 40 --seed 47 --classes artifacts/classes-frozen-canonical.npz \
  --allow-remote-code --device cpu --output artifacts/real-packshots-frozen-v2
python3 -m wineid.fusion_run --candidate-images artifacts/tz/images.csv --uploads "$UPLOADS" \
  --sample 40 --seed 47 --classes artifacts/classes-lora-canonical.npz \
  --adapter artifacts/lora/jina-clip-v2-lora-synthetic.pt \
  --allow-remote-code --device cpu --output artifacts/real-packshots-lora-v2
python3 -m wineid.fusion_compare artifacts/real-packshots-frozen-v2.report.json \
  artifacts/real-packshots-lora-v2.report.json
```

Быстрый опциональный сквозной тест HTTP с этими же весами и настоящим фото (без OCR): `WINE_SMOKE_JINA=1 WINE_ALLOW_REMOTE_CODE=1 python3 -m pytest -q tests/test_smoke_vision.py`. Проверено: **1 passed**; весь обычный suite без весов/платных вызовов — **188 passed, 5 skipped**. На данном macOS окружении предварительно нужен OpenMP workaround ниже.

При повторе выбрать **новый префикс** (вывод не перезаписывается). Артефакты лежат в игнорируемом git каталоге `artifacts/`; их нужно сохранить отдельно для повторения на другой машине. Для `--manifest <queries.jsonl>` запускается `checked_queries` (GT, split, SHA и gallery-leakage) **до** загрузки весов и выводится `fusion_eval` по указанному split; предоставление независимого GT и проверка `gt_source` остаются обязанностью владельца данных. Для разрешённого OCR: `--ocr live --allow-live-ocr --roi-mode center_80_crop` и `MISTRAL_API_KEY` (эвристический crop **не** найденная этикетка), либо предоставить ориентированные bbox в manifest и оставить `--roi-mode manual`. Предварительно согласовать передачу изображений/расходы и проверить реальный API Mistral. `--ocr manual` использует только `manual_text` из manifest и не измеряет качество OCR. Без явного выбора OCR **нет сетевых вызовов к провайдеру**. Evidence JSONL при включённом OCR содержит нормализованный распознанный текст — хранить как чувствительные данные, не публиковать. Общий deadline пока мягкий: синхронный provider HTTP-вызов нельзя гарантированно прервать при зависании DNS/сокета; CLIP inference теперь изолирован и прерывается, но parent preprocessing/скоринг не доказан жёстко ограниченным. Это блокер production.

В **данном смешанном macOS conda/user-site окружении** одновременно грузились разные `libomp.dylib` от torch и conda SciPy и процесс аварийно завершался. Прогоны выше выполнялись также с `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0` и `DYLD_LIBRARY_PATH=/Users/eu/.local/lib/python3.12/site-packages/torch/lib` (только workaround этого окружения). Предпочтительно изолированное окружение с одним OpenMP runtime; **не** включать `KMP_DUPLICATE_LIB_OK=TRUE` как «исправление». На другой машине `DYLD_LIBRARY_PATH` может сломать загрузку, применять только после диагностики. При загрузке модель вывела предупреждения о транзитивном remote code и regex токенизатора; полного lock/аудита этих зависимостей нет, даже если основной commit Jina закреплён.

## Зафиксированное наблюдение (не качество на независимом GT)

- На 3 `eval/` изображениях создан `artifacts/real-unlabelled-fusion-lora.report.json`: 3/3 получены class-кандидаты, без OCR; никаких метрик accuracy. Top-1 по порядку входных файлов: `kuban-vino-shato-tamane-rezerv-myuller-turgau-limited-edishn-2021-beloe-suhoe-12`, `aristov-anima-millesimato-beloe-bryut`, `massandra-muskat`. Основной API с настоящими весами возвращает `ambiguous`/`slug=null` без policy; `/v1/eval/predict` возвращает принудительный slug после исправления обработки слишком большого OCR-кропа. Совпадающие class prompts (116 SKU в 55 группах для canonical) не могут привести к принятому slug на одном class-сигнале даже с явно заданной policy; eval по-прежнему выдаёт best-effort top-1.
- На детерминированной выборке 40 настоящих **студийных** uploads по 40 разным производителям, с эвристическими CSV↔uploads связями: совпадение с **кандидатным** slug Top-1 frozen 4/40, LoRA 5/40; Top-5 8/40 и 11/40. Top-1 изменился на 31/40, LoRA помог по кандидатной ссылке в 2 случаях и повредил в 1. Это **не** independent test и **не** полевая точность, по этим числам нельзя утверждать выигрыш адаптера, подбирать веса или выпускать policy. CPU p95 на этих 40 запросах (после загрузки модели, последовательные запросы): frozen ~1494 мс, LoRA ~1540 мс; не end-to-end серверный SLA.
- Сравнение файлов `artifacts/real-packshots-{frozen,lora}-v2.report.json` проверяет одинаковые байты/порядок снимков, версию каталога и prompt/scoring. `images.csv` checksum `b55475d51f8181ebe9bff7fb3464926b11b2d6e53593616c70fb81eea959eed1`; class indexes SHA256: frozen `b0513f37ce545d3001ed1f672256c8a677b228b84d4ec861f033b46dcba15f34`, LoRA `509f6075b9454bbd6fbd14f2c5ddc24666cb8a889fdd435aca3badbfb27ea653`.

## Не закрыто (реальные gate, а не TODO в коде)

1. Ручное подтверждение соответствий локальных uploads и CSV, provenance и независимая разметка пользовательских фото, включая unknown/похожие SKU, source_group split до настройки; `eval/` не использовать для подбора.
2. Реальный OCR canary только с ключом/согласием и ручными ROI; сравнение на одних запросах OCR-only, frozen/LoRA, union, RRF/gap и family. Воспроизводимый offline harness готов, но качественный OCR здесь не измерен.
3. Калибровать пороги/отказы только на real validation, заморозить версии и оценить untouched test. Пока основной API должен возвращать кандидатов без принятого slug. Лицензии, ограничения удалённого провайдера, full-vision веса и end-to-end p95 с очередью — открытые блокеры выпуска.
