#!/usr/bin/env bash
# Прогон финального датасета организаторов через наш сервис.
#
#   bash ./run.sh <папка_датасета> [файл_результата]
#
# <папка_датасета> — распакованный пакет организатора: queries/, queries.tsv,
# participant_test.sh (если скрипта в пакете нет, берётся eval/participant_test.sh).
# Результат по умолчанию: results/<имя_папки>-predictions.jsonl.
#
# Что делает: читает .env (если есть), выставляет переменные сервиса, при
# необходимости собирает индекс классов CLIP, поднимает API на 127.0.0.1:8080,
# ждёт готовности распознавания, запускает runner организаторов и гасит API.
#
# Переменные (можно задать в окружении или в .env, см. .env.example):
#   MINERU_TOKEN        ключ облачного OCR MinerU; без него работает только CLIP
#   WINE_USE_OCR        1 — включить OCR MinerU (по умолчанию 0: облачный OCR
#                       отвечает дольше 10 с лимита runner, ответы уходили бы в null)
#   PYTHON              интерпретатор с зависимостями (по умолчанию .venv)
# Нужны bash, curl, jq, awk, sha256sum/shasum и Python 3.12 с
# requirements-dev.txt + requirements-clip.txt.
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
cd "$root"

die() { printf 'run.sh: %s\n' "$*" >&2; exit 1; }

[ "$#" -ge 1 ] || die "usage: bash ./run.sh <папка_датасета> [файл_результата]"
dataset=$(cd "$1" && pwd) || die "нет папки датасета: $1"
[ -f "$dataset/queries.tsv" ] || die "в $dataset нет queries.tsv"
[ -d "$dataset/queries" ] || die "в $dataset нет папки queries/"
output=${2:-"results/$(basename "$dataset")-predictions.jsonl"}
case "$output" in /*|[A-Za-z]:*) ;; *) output="$root/$output" ;; esac
[ ! -e "$output" ] || die "результат уже существует: $output (переименуйте или укажите другой)"

for tool in curl jq awk; do
  command -v "$tool" >/dev/null 2>&1 || die "не найден $tool"
done

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

if [ -z "${PYTHON:-}" ]; then
  for candidate in .venv/bin/python .venv/Scripts/python.exe python3 python; do
    if [ -x "$candidate" ] || command -v "$candidate" >/dev/null 2>&1; then PYTHON=$candidate; break; fi
  done
fi
[ -n "${PYTHON:-}" ] || die "не найден Python"

export WINE_HOST=127.0.0.1
export WINE_PORT=${WINE_PORT:-8080}
export WINE_ROI_MODE=${WINE_ROI_MODE:-center_80_crop}
export WINE_CLASS_INDEX=${WINE_CLASS_INDEX:-artifacts/classes-lora-canonical.npz}
export WINE_ADAPTER=${WINE_ADAPTER:-artifacts/lora/jina-clip-v2-lora-synthetic.pt}
export WINE_ALLOW_REMOTE_CODE=${WINE_ALLOW_REMOTE_CODE:-1}
# Runner организаторов ждёт ответ не дольше 10 с; бюджет сервиса чуть меньше.
export WINE_REQUEST_TIMEOUT=${WINE_REQUEST_TIMEOUT:-9}
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
MINERU_TOKEN=${MINERU_TOKEN:-${MINERU_API_KEY:-}}
if [ "${WINE_USE_OCR:-0}" = "1" ] && [ -n "$MINERU_TOKEN" ]; then
  export MINERU_TOKEN WINE_OCR_PROVIDER=mineru
else
  export WINE_OCR_PROVIDER=none
fi

[ -f "$WINE_ADAPTER" ] || die "нет весов LoRA: $WINE_ADAPTER"
if [ ! -f "$WINE_CLASS_INDEX" ]; then
  echo "== индекс классов не найден, собираю $WINE_CLASS_INDEX (один раз)"
  "$PYTHON" -m wineid.clip_zero_shot build-classes "$WINE_CLASS_INDEX" \
    --csv strapi_output0709.csv --prompt canonical \
    --adapter "$WINE_ADAPTER" --allow-remote-code
fi

mkdir -p results run-logs
log="run-logs/api-$(date +%Y%m%d-%H%M%S).log"
echo "== OCR: $WINE_OCR_PROVIDER, ROI: $WINE_ROI_MODE, бюджет запроса: ${WINE_REQUEST_TIMEOUT} с"
echo "== запускаю API (лог: $log)"
"$PYTHON" -m wineid.api >"$log" 2>&1 &
api_pid=$!
cleanup() {
  # Git Bash: venv python.exe is a launcher with a child process, stop the whole tree.
  if [ -r "/proc/$api_pid/winpid" ] && command -v taskkill >/dev/null 2>&1; then
    taskkill //F //T //PID "$(cat "/proc/$api_pid/winpid")" >/dev/null 2>&1 || true
  fi
  kill "$api_pid" 2>/dev/null || true
  wait "$api_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

base="http://127.0.0.1:$WINE_PORT"
for _ in $(seq 1 300); do
  kill -0 "$api_pid" 2>/dev/null || { tail -n 30 "$log" >&2; die "API завершился при запуске"; }
  if curl -fsS --max-time 2 "$base/api/health/recognition" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS --max-time 2 "$base/api/health/recognition" >/dev/null 2>&1 || { tail -n 30 "$log" >&2; die "распознавание не готово за 300 с"; }
echo "== API готов: $(curl -fsS "$base/api/health" | jq -c '{wine_count, recognition_ready, ocr_configured, roi_mode}')"

runner="$dataset/participant_test.sh"
[ -f "$runner" ] || runner="$root/eval/participant_test.sh"
echo "== прогон: $runner"
mkdir -p "$(dirname "$output")"
( cd "$dataset" && bash "$runner" \
    --images-dir ./queries \
    --manifest ./queries.tsv \
    --endpoint "$base/v1/eval/predict" \
    --output "$output" )

total=$(wc -l <"$output" | tr -d ' ')
nulls=$(jq -s '[.[] | select(.predicted_slug == null)] | length' "$output")
p50=$(jq -s '[.[].latency_ms] | sort | .[length/2|floor]' "$output")
pmax=$(jq -s '[.[].latency_ms] | max' "$output")
echo "== готово: $output"
echo "   строк: $total, без ответа (null): $nulls, latency p50: ${p50} мс, max: ${pmax} мс"
