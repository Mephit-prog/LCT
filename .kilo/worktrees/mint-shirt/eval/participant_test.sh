#!/usr/bin/env bash

# Sends every image from queries.tsv to the participant service and writes only
# query_id, image_path, image_sha256, predicted_slug and latency_ms to JSONL.

set -uo pipefail

images_dir=""
manifest=""
endpoint="http://127.0.0.1:8080/v1/eval/predict"
output="predictions.jsonl"

usage() {
  printf '%s\n' \
    "Usage: $0 --images-dir DIR --manifest FILE [--endpoint URL] [--output FILE]" \
    "" \
    "Manifest header: query_id<TAB>image_path" \
    "Accepted responses: {\"slug\":\"...\"} or [{\"slug\":\"...\"}]"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --images-dir)
      [ "$#" -ge 2 ] || die "--images-dir requires a value"
      images_dir="$2"
      shift 2
      ;;
    --manifest)
      [ "$#" -ge 2 ] || die "--manifest requires a value"
      manifest="$2"
      shift 2
      ;;
    --endpoint)
      [ "$#" -ge 2 ] || die "--endpoint requires a value"
      endpoint="$2"
      shift 2
      ;;
    --output)
      [ "$#" -ge 2 ] || die "--output requires a value"
      output="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

for command_name in curl jq awk mktemp; do
  command -v "$command_name" >/dev/null 2>&1 || \
    die "required command not found: $command_name"
done

[ -n "$images_dir" ] || die "--images-dir is required"
[ -d "$images_dir" ] || die "images directory not found: $images_dir"
[ -n "$manifest" ] || die "--manifest is required"
[ -f "$manifest" ] || die "manifest not found: $manifest"
[ -n "$endpoint" ] || die "endpoint must not be empty"
[ ! -e "$output" ] || die "output already exists: $output"

if ! awk -F '\t' '
  NR == 1 {
    sub(/\r$/, "", $2)
    if (NF != 2 || $1 != "query_id" || $2 != "image_path") bad = 1
    next
  }
  {
    sub(/\r$/, "", $2)
    if (NF != 2 || $1 !~ /^[A-Za-z0-9._-]+$/ || $2 == "" || seen[$1]++) bad = 1
    count++
  }
  END { exit (bad || count == 0) }
' "$manifest"; then
  die "manifest must contain unique query_id<TAB>image_path rows"
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha256_file() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  die "required command not found: sha256sum or shasum"
fi

work_dir=$(mktemp -d) || die "cannot create temporary directory"
case "$work_dir" in
  /tmp/*|/private/tmp/*|/var/tmp/*|/var/folders/*|/private/var/folders/*) ;;
  *) die "unexpected temporary directory: $work_dir" ;;
esac

cleanup() {
  rm -rf -- "$work_dir"
}
trap cleanup EXIT HUP INT TERM

temporary_output="$work_dir/predictions.jsonl"
response_body="$work_dir/response.json"
: > "$temporary_output"

exec 3< "$manifest"
IFS= read -r ignored_header <&3 || die "cannot read manifest"

while IFS=$'\t' read -r query_id image_relpath extra <&3 || \
  [ -n "${query_id:-}${image_relpath:-}${extra:-}" ]; do
  image_relpath=${image_relpath%$'\r'}

  case "$image_relpath" in
    /*|..|../*|*/..|*/../*) die "unsafe image_path: $image_relpath" ;;
  esac

  image_file="$images_dir/$image_relpath"
  [ -f "$image_file" ] || die "image not found: $image_relpath"

  image_sha256=$(sha256_file "$image_file") || \
    die "cannot calculate SHA-256: $image_relpath"

  curl_file=$image_file
  curl_file=${curl_file//\\/\\\\}
  curl_file=${curl_file//\"/\\\"}
  form_value="image=@\"${curl_file}\""

  curl_meta=$(curl \
    --silent \
    --show-error \
    --request POST \
    --connect-timeout 5 \
    --max-time 10 \
    --output "$response_body" \
    --write-out $'%{http_code}\t%{time_total}' \
    --form "$form_value" \
    "$endpoint")
  curl_result=$?

  http_code=${curl_meta%%$'\t'*}
  time_total=${curl_meta#*$'\t'}
  case "$time_total" in
    ''|*[!0-9.]*) latency_ms=0 ;;
    *) latency_ms=$(awk -v seconds="$time_total" 'BEGIN { printf "%.0f", seconds * 1000 }') ;;
  esac

  predicted_slug=""
  if [ "$curl_result" -eq 0 ] && { [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; }; then
    predicted_slug=$(jq -er '
      if type == "object" then .slug
      elif type == "array" and length > 0 then .[0].slug
      else empty
      end
      | select(type == "string" and length > 0)
    ' "$response_body" 2>/dev/null) || predicted_slug=""
  fi

  jq -cn \
    --arg query_id "$query_id" \
    --arg image_path "$image_relpath" \
    --arg image_sha256 "$image_sha256" \
    --arg predicted_slug "$predicted_slug" \
    --argjson latency_ms "$latency_ms" \
    '{
      query_id: $query_id,
      image_path: $image_path,
      image_sha256: $image_sha256,
      predicted_slug: (if $predicted_slug == "" then null else $predicted_slug end),
      latency_ms: $latency_ms
    }' >> "$temporary_output" || die "cannot write result: $query_id"
done

exec 3<&-
mkdir -p "$(dirname "$output")" || die "cannot create output directory"
mv "$temporary_output" "$output" || die "cannot write output: $output"
