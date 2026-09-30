# Инструкция по запуску проекта (Фронтенд + Бэкенд)

Эта инструкция описывает, как локально запустить бэкенд и фронтенд для проекта **LCT (Wine ID / Recognition)**.

---

## Предварительные требования
- **Python** (версии 3.10 или выше)
- **Node.js** (версии 18+ и npm)
- Установленные зависимости (модели, LoRA адаптеры и классификаторы) в папке `backend/artifacts`.

Эта инструкция относится к собранному Windows-комплекту, а не к чистому клону.
Подготовка: `python build_windows_bundle.py D:\LCT --provision` на Windows;
ограничения поставки описаны в [DELIVERY_STATUS.md](DELIVERY_STATUS.md).
Веса и индексы не входят в Git. Токен получите у провайдера и не коммитьте;
ранее опубликованный токен необходимо отозвать.

---

## 1. Запуск бэкенда (FastAPI)

1. Откройте терминал и перейдите в папку `backend`:
   ```powershell
   cd D:\LCT\backend
   ```

2. Установите необходимые переменные окружения (в PowerShell):
   ```powershell
   $env:WINE_ROI_MODE="center_80_crop"
   $env:WINE_CLASS_INDEX="artifacts/classes-lora-canonical.npz"
   $env:WINE_ADAPTER="artifacts/lora/jina-clip-v2-lora-synthetic.pt"
   $env:WINE_ALLOW_REMOTE_CODE="1"
   $env:WINE_OCR_PROVIDER="mineru"
   $env:MINERU_TOKEN="<ваш токен MinerU>"
   $env:WINE_REQUEST_TIMEOUT="60.0"
   ```

3. Запустите виртуальное окружение и API-сервер:
   ```powershell
   .venv\Scripts\python -m wineid.api
   ```

> Бэкенд запустится на **`http://127.0.0.1:8080`**.

---

## 2. Запуск фронтенда (Vue / Vite)

1. Откройте **новый** терминал и перейдите в папку `web`:
   ```powershell
   cd D:\LCT\web
   ```

2. *(Опционально)* Убедитесь, что в папке `web` задан увеличенный таймаут запросов (так как обработка фото с OCR может занимать 15–25 секунд). В файле `web/.env.local` должно быть:
   ```env
   VITE_API_BASE_URL=/api
   VITE_SCAN_TIMEOUT_MS=30000
   VITE_SOMMELIER_TIMEOUT_MS=15000
   ```

3. Запустите сервер разработки Vite:
   ```powershell
   npm run dev
   ```

> Фронтенд запустится на **`http://localhost:5173/`**.  
> Все запросы к `/api` автоматически проксируются Vite на бэкенд (`http://127.0.0.1:8080`).

---

## Проверка работы
1. Откройте в браузере **`http://localhost:5173/`**.
2. Загрузите или сфотографируйте этикетку вина.
3. Дождитесь завершения распознавания (поскольку OCR и анализ модели занимают ~15-20 секунд, убедитесь, что не закрываете страницу раньше времени).
