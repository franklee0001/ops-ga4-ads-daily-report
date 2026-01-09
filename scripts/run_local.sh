#!/usr/bin/env bash
set -euo pipefail

export TZ=Asia/Seoul

if [ -f ".env" ]; then
  set -a
  . "./.env"
  set +a
fi

if [ -z "${PROPERTY_ID:-}" ] || [ -z "${CUSTOMER_ID:-}" ]; then
  echo "오류: PROPERTY_ID와 CUSTOMER_ID가 필요합니다. .env를 확인해주세요."
  exit 1
fi

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

# shellcheck disable=SC1091
. ".venv/bin/activate"

python -m pip install -U pip
python -m pip install -r requirements.txt

python src/smoke_test.py
python src/daily_report.py

today="$(date +%F)"
report_dir="reports/${today}"
index_path="${report_dir}/index.html"

errors=0

if [ ! -d "${report_dir}" ]; then
  echo "오류: 리포트 폴더가 없습니다: ${report_dir}"
  errors=1
fi

if [ ! -f "${index_path}" ]; then
  echo "오류: 리포트 index.html이 없습니다: ${index_path}"
  errors=1
fi

png_files=""
while IFS= read -r -d '' png; do
  png_files="${png_files}${png}"$'\n'
done < <(find "${report_dir}" -maxdepth 1 -type f -name "*.png" -print0 2>/dev/null)

if [ -z "${png_files}" ]; then
  echo "오류: PNG 차트가 없습니다: ${report_dir}"
  errors=1
fi

if [ -f "${index_path}" ] && [ -n "${png_files}" ]; then
  missing_refs=0
  while IFS= read -r png; do
    [ -z "${png}" ] && continue
    name="$(basename "${png}")"
    if ! grep -Fq "${name}" "${index_path}"; then
      echo "오류: index.html에서 PNG 참조 누락: ${name}"
      missing_refs=1
    fi
  done <<EOF_PNG
${png_files}
EOF_PNG

  if [ "${missing_refs}" -ne 0 ]; then
    errors=1
  fi
fi

if [ "${errors}" -ne 0 ]; then
  echo "검증 실패. 상세 오류를 확인해주세요."
  exit 1
fi

echo "검증 완료:"
echo "  리포트 폴더: ${report_dir}"
echo "  리포트 파일: ${index_path}"

if command -v open >/dev/null 2>&1; then
  open "${index_path}"
fi
