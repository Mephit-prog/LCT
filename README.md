# LCT · Сканер российских вин

Исследовательский прототип поиска карточки вина по фотографии этикетки для платформы «Своё Вино». Backend — Python/FastAPI, frontend — Vue/Vite. Независимые метрики качества и запуск полного комплекта на целевой Windows-машине пока не подтверждены.

## Структура

- `wineid/` — API, поиск, OCR, CLIP, оценка и сборка артефактов.
- `web/` — интерфейс сканирования и карточки вина.
- `tests/` — backend unit/regression и opt-in smoke-тесты.
- `eval/` — три неразмеченных примера для проверки интерфейса, не validation-набор.
- `virtual_somelie/` — отдельный исследовательский модуль, не подключён к API.
- `local/run-local.ps1` — управление локальным запуском на Windows.
- `strapi_output0709.csv`, `producer_aliases.txt`, `artifacts/tz/` — данные каталога и справочники.

## Прогон на датасете организаторов

Нужны Python 3.12, `bash`, `curl`, `jq`, `awk` (на Windows — Git Bash). Сетап один раз:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130   # Linux: .venv/bin/python
.venv/Scripts/python -m pip install -c constraints.txt -r requirements-dev.txt -r requirements-clip.txt
```

Без GPU ставьте обычный `torch` из PyPI. Запуск на распакованном пакете организатора (`queries/`, `queries.tsv`, `participant_test.sh`):

```bash
bash ./run.sh <папка_датасета>
```

`run.sh` поднимает API на `127.0.0.1:8080` (CLIP Jina v2 + LoRA `artifacts/lora/jina-clip-v2-lora-synthetic.pt`, индекс `artifacts/classes-lora-canonical.npz`), ждёт готовности, запускает runner организаторов и пишет `results/<папка>-predictions.jsonl`. При первом запуске веса Jina скачиваются с Hugging Face. Облачный OCR MinerU по умолчанию выключен: он не укладывается в 10-секундный лимит runner; включается `WINE_USE_OCR=1` и `MINERU_TOKEN` в `.env` (шаблон — `.env.example`).

Результат финального датасета: [results/final-predictions.jsonl](results/final-predictions.jsonl), 100/100 ответов, p50 0,7 с.

## Локальная разработка

Для проверок используйте Python 3.12 и Node.js 18+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints.txt -r requirements-dev.txt
python -m wineid.api
```

На Windows активация: `.venv\Scripts\Activate.ps1`.
API: `http://127.0.0.1:8080`. В отдельном терминале:

```bash
cd web
npm ci
npm run dev
```

UI: `http://localhost:5173`; Vite проксирует `/api` на backend.
Это минимальное окружение без модельных весов и внешнего OCR, не полноценное распознавание. Для модельного пути нужны совместимые веса, индекс классов и явная конфигурация; см. [JINA_CLIP.md](JINA_CLIP.md).

## Проверки

```bash
python -m pip check
python -m compileall -q wineid tests
python -m pyflakes wineid tests
python -m pytest tests -q -rs
```

```bash
cd web
npm test
npm run type-check
npm run build
```

Математический модуль проверяется в отдельном окружении: `pip install -r requirements-somelie.txt`, затем `python -m pytest virtual_somelie -q`. Облачные и тяжёлые модельные smoke-тесты требуют явного opt-in, ресурсов и разрешения на использование API.

## Документация и ограничения

- [API.md](API.md) — HTTP и CLI-контракты.
- [RUN_GUIDE.md](RUN_GUIDE.md) — запуск собранного Windows-комплекта.
- [DELIVERY_STATUS.md](DELIVERY_STATUS.md) — текущие ограничения поставки.
- [MINERU_CLOUD.md](MINERU_CLOUD.md) — opt-in облачный OCR.
- [LICENSES.md](LICENSES.md) — лицензии и ограничения использования.
- [web/README.md](web/README.md), [virtual_somelie/README.md](virtual_somelie/README.md) — документация модулей.

Веса, индексы и HF-кэш не входят в Git. Синтетические эксперименты и три фото `eval/` не подтверждают production-качество. Токены передаются через окружение и не должны попадать в исходники или комплект поставки. Ранее опубликованный токен требуется отозвать: удаление из инструкции не удаляет его из истории Git.

Исторические планы, материалы верификации и копии agent worktrees перемещены в `.temp/` на рабочей машине. Эта директория игнорируется Git и не нужна для сборки или ревью.
