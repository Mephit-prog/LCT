# Статус поставки

`RUN_GUIDE.md` не изменён. Исходный код бэкенда один: `wineid/`.
На Windows с Python и Node.js 18+ выполните перед передачей комплекта:

```powershell
python build_windows_bundle.py D:\LCT --provision
```

Команда собирает `D:\LCT\backend` (веса копируются из `artifacts/`), `web`, `local`, устанавливает backend `.venv` и `web/node_modules` без второй поддерживаемой копии кода. Сборщик загружает pinned HF-веса/code в `backend/artifacts/hf-cache` и проверяет повторную загрузку offline; для этого нужны сеть, достаточно RAM и согласованная лицензия. **Windows-сборка и запуск пока не подтверждены на целевой машине.** Фронтенд переведён на Vite 5; `npm ci` и тесты проверены на Node.js 18.20.8.

MinerU cloud обрабатывает изображения при `WINE_OCR_PROVIDER=mineru` и `MINERU_TOKEN`; `mineru_local` — прежний локальный CLI. Cloud OCR не разрешает policy accepted. Ключ в RUN_GUIDE.md оставлен по условию; его квота и срок требуют согласования с организаторами.

Остаются до готовности: реальный Windows smoke (CLIP + MinerU, валидный конкурсный ключ и целевые веса), измерения p50/p95, перегрузка и независимый validation/test качества. Интеграционный облачный smoke нельзя безопасно выполнять в CI без разрешения и квоты. Текущий frontend скрывает отправку заявки и сомелье, пока backend не реализует эти операции.
