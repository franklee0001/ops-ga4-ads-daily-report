#!/usr/bin/env bash
set -euo pipefail

export TZ=Asia/Seoul

target_date="${1:-$(date +%F)}"
report_dir="reports/${target_date}"
index_path="${report_dir}/index.html"

printf "리포트 검증: %s\n" "${target_date}"
printf "폴더: %s\n" "${report_dir}"

if [ -d "${report_dir}" ]; then
  echo "폴더 존재: OK"
else
  echo "폴더 존재: 실패"
  echo "가능한 원인: 리포트가 아직 생성되지 않았거나 날짜가 다릅니다."
  echo "확인 위치: scripts/run_local.sh, src/daily_report.py"
fi

if [ -f "${index_path}" ]; then
  echo "HTML 존재: OK"
else
  echo "HTML 존재: 실패"
  echo "가능한 원인: 템플릿 렌더링 실패 또는 출력 경로 오류입니다."
  echo "확인 위치: src/daily_report.py, templates/report_ko.html"
fi

echo ""
echo "PNG 파일 목록:"
png_files=""
while IFS= read -r -d '' png; do
  png_files="${png_files}${png}"$'\n'
done < <(find "${report_dir}" -maxdepth 1 -type f -name "*.png" -print0 2>/dev/null || true)

if [ -n "${png_files}" ]; then
  printf "%s\n" "${png_files}"
else
  echo "(없음)"
  echo "가능한 원인: 차트 생성 로직이 실행되지 않았거나 데이터가 비어 있습니다."
  echo "확인 위치: src/daily_report.py (generate_charts/_plot_chart)"
fi

echo ""
echo "HTML 내 이미지 참조 (상위 20개):"
if [ -f "${index_path}" ]; then
  if command -v rg >/dev/null 2>&1; then
    rg -o 'src="[^"]+\.png[^"]*"' "${index_path}" | head -n 20 || true
  else
    grep -Eo 'src="[^"]+\.png[^"]*"' "${index_path}" | head -n 20 || true
  fi
else
  echo "(HTML 없음)"
fi

if [ -f "${index_path}" ] && [ -n "${png_files}" ]; then
  missing_refs=0
  while IFS= read -r png; do
    [ -z "${png}" ] && continue
    name="$(basename "${png}")"
    if ! grep -Fq "${name}" "${index_path}"; then
      missing_refs=1
      echo "누락 참조: ${name}"
    fi
  done <<EOF_PNG
${png_files}
EOF_PNG

  if [ "${missing_refs}" -ne 0 ]; then
    echo "가능한 원인: 템플릿 경로 프리픽스 설정이 잘못되었습니다."
    echo "확인 위치: src/daily_report.py (build_render_context)"
  fi
fi
