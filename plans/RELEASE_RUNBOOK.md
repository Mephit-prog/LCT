# Черновик runbook для wineid (не RC)

Этот runbook применяется **только** к wine HTTP. Облачный MinerU Precision
([MINERU_CLOUD.md](../MINERU_CLOUD.md)) имеет отдельные БД, секреты, egress,
воркер и callback; его readiness/квоты не должны переключать `/api/recognize`.

## Подготовка (выполняет оператор на выбранной платформе)

1. Утвердить лицензии и версии Python/OS/torch/HF remote code, локального OCR и
   весов. Собрать *полные* locks API, CLIP, OCR и immutable образ/commit;
   `constraints.txt` — лишь ограничения зависимостей, не lock. Не загружать
   веса и код при обработке пользовательского запроса. Сохранить предыдущий
   рабочий bundle отдельно для отката.
2. Заполнить приватный read-only каталог согласованными CSV, aliases, индексами,
   checkpoint, OCR manifest/весами и policy. Использовать
   `python -m wineid.release_bundle build ...` / `verify ...` по
   [API.md](../API.md); перечислить *все* выбранные роли, lock-файлы,
   закрепить и подписать manifest во внешнем доверенном канале. Не включать
   токены, фото, OCR-текст в bundle. `WINE_RELEASE_MANIFEST` проверяет файлы и
   пути до инициализации моделей, но не удостоверяет автора manifest.
3. Настроить один API-процесс, лимиты CPU/RAM/VRAM/FD/temp, TLS и proxy с
   лимитами тела, количества соединений и времени upload. Проверить отдельный
   `WINE_TOKEN` в secret store; включить только согласованный локальный OCR.
   `WINE_ROI_MODE=refuse` — безопасный default, не менять без policy. Многокопийный
   CLIP и warm Paddle без общего планировщика не включать.

## Canary, проверка и решение об открытии трафика

- Из чистого окружения: `python -m pip check`, `python -m compileall -q wineid tests`,
  `python -m pyflakes wineid tests`, `python -m pytest -q -rs`.
  Отдельно с разрешёнными данными и *реальными* закреплёнными весами:
  `WINE_SMOKE_OFFLINE=1 python -m pytest -q tests/test_smoke_offline.py`,
  `WINE_SMOKE_JINA=1 WINE_ALLOW_REMOTE_CODE=1 python -m pytest -q tests/test_smoke_vision.py`,
  локальный OCR canary и сквозной HTTP→ROI→OCR/CLIP→fusion. Сохранять команды,
  версии и версии артефактов, а не payload/секреты. Пропуск smoke — блокер.
- `GET /ready` проверяет **только каталог**; `GET /api/health/recognition`
  должен дать 200 для распознавания, а `acceptance_ready` — true только при
  согласованной policy. Если её нет — оставить candidates-only/отказы.
  Eval `/v1/eval/predict` не проверяет production acceptance.
- На согласованной одновременной нагрузке измерить p50/p95/p99 (включая upload),
  RSS/VRAM, процессы/FD, накопление temp, повторное восстановление OCR/CLIP,
  отказ при превышении лимита. Требования по нагрузке и точности согласовать
  **до** прогона; CLIP inference прерывается через kill process group,
  но parent preprocessing/scoring и OCR ещё не подтверждены под общим жёстким
  deadline. До проверки на target и reverse proxy не объявлять wall-clock SLA.
- Только после независимого validation/test с known/unknown/ambiguous,
  provenance и аудитом утечек обсуждать авто-принятие slug или RC. Результаты
  синтетики/кандидатных связей uploads не заменяют этот gate.

## Сбой и rollback

При несовместимой policy/весах, падении canary, росте задержки/ошибок или
несовпадении SHA: прекратить приём запросов распознавания на proxy, сохранить
только обезличенные агрегаты и диагностические коды; остановить API (воркеры
OCR/CLIP закрываются через runtime lifecycle), вернуть предыдущие *совместимые*
образ+bundle+policy единым переключением. Проверить `verify`, canary и
`/api/health/recognition` перед возвращением трафика. Изменение ROI/scoring или
весов требует новой проверки policy. Фактический тест rollback на целевом
сервере ещё не выполнен.
