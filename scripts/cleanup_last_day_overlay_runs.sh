#!/usr/bin/env bash
set -euo pipefail

# Cleanup utility for recent optimize_overlay runs.
#
# What this script does:
# 1) Stops optimize_overlay processes launched from this repo (with child workers).
# 2) Deletes only last-day overlay run artifacts/logs (or configurable hours via --since-hours):
#    - artifacts/reports/*fetch_window_failed*.jsonl
#    - artifacts/reports/overlay_ultra_fetch_failures_*.jsonl
#    - artifacts/reports/overlay_ultra_crash_*.txt
#    - artifacts/reports/overlay_ultra_partial_*.json{,l}
#    - artifacts/execution_overlay/**/<RUN_ID> directories newer than the cutoff
#    - logs/ultra_*.log newer than the cutoff
#    - artifacts/**/runs/* and artifacts/**/checkpoints/* entries that contain run IDs
#      detected from recent overlay files/logs (e.g. 20260210_204206)
#
# Safety:
# - Never touches data/processed/** cache data.
# - Only matches optimize_overlay processes whose command line references this repo root.
# - Supports --dry-run to print actions without deleting.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0
SINCE_HOURS=24

usage() {
  cat <<'EOF'
Usage:
  scripts/cleanup_last_day_overlay_runs.sh [--dry-run] [--since-hours N]

Examples:
  scripts/cleanup_last_day_overlay_runs.sh
  scripts/cleanup_last_day_overlay_runs.sh --dry-run
  scripts/cleanup_last_day_overlay_runs.sh --since-hours 12
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --since-hours)
      SINCE_HOURS="${2:-}"
      if [[ -z "${SINCE_HOURS}" || ! "${SINCE_HOURS}" =~ ^[0-9]+$ ]]; then
        echo "Invalid --since-hours value: ${SINCE_HOURS}" >&2
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SINCE_EXPR="${SINCE_HOURS} hours ago"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

_self_pgid() {
  ps -o pgid= -p "$$" | tr -d ' '
}

_find_overlay_matches() {
  ps -eo pid=,ppid=,pgid=,args= | awk -v root="${REPO_ROOT}" -v self_pid="$$" '
    {
      pid=$1; ppid=$2; pgid=$3;
      $1=""; $2=""; $3="";
      sub(/^[[:space:]]+/, "", $0);
      has_target = ($0 ~ /(^|[[:space:]\/])optimize_overlay\.py([[:space:]]|$)/) || ($0 ~ /(^|[[:space:]])-m[[:space:]]+bot087\.cli\.optimize_overlay([[:space:]]|$)/);
      in_repo = (index($0, root) > 0);
      if (pid != self_pid && has_target && in_repo) {
        printf "%s|%s|%s|%s\n", pid, ppid, pgid, $0;
      }
    }
  '
}

_job_alive() {
  local pid="$1"
  local pgid="$2"
  local own_pgid="$3"
  if kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  if [[ -n "${pgid}" && "${pgid}" != "${own_pgid}" ]]; then
    if ps -eo pgid= | awk -v g="${pgid}" '$1==g{found=1; exit 0} END{exit(found?0:1)}'; then
      return 0
    fi
  fi
  return 1
}

_signal_target() {
  local sig="$1"
  local pid="$2"
  local pgid="$3"
  local own_pgid="$4"
  kill "-${sig}" "${pid}" 2>/dev/null || true
  pkill "-${sig}" -P "${pid}" 2>/dev/null || true
  if [[ -n "${pgid}" && "${pgid}" != "${own_pgid}" ]]; then
    kill "-${sig}" -- "-${pgid}" 2>/dev/null || true
  fi
}

_wait_for_stop() {
  local timeout_sec="$1"
  local pid="$2"
  local pgid="$3"
  local own_pgid="$4"
  local deadline=$((SECONDS + timeout_sec))
  while ((SECONDS < deadline)); do
    if ! _job_alive "${pid}" "${pgid}" "${own_pgid}"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

stop_overlay_processes() {
  local -a matches=()
  local -a roots=()
  local line pid ppid pgid cmd own_pgid
  declare -A pid_set=()

  mapfile -t matches < <(_find_overlay_matches)
  if ((${#matches[@]} == 0)); then
    log "No optimize_overlay process found for repo ${REPO_ROOT}."
    return 0
  fi

  for line in "${matches[@]}"; do
    IFS='|' read -r pid ppid pgid cmd <<< "${line}"
    pid_set["${pid}"]=1
  done
  for line in "${matches[@]}"; do
    IFS='|' read -r pid ppid pgid cmd <<< "${line}"
    if [[ -z "${pid_set[${ppid}]:-}" ]]; then
      roots+=("${line}")
    fi
  done
  if ((${#roots[@]} == 0)); then
    roots=("${matches[@]}")
  fi

  own_pgid="$(_self_pgid)"
  for line in "${roots[@]}"; do
    IFS='|' read -r pid ppid pgid cmd <<< "${line}"
    log "Stopping optimize_overlay pid=${pid} ppid=${ppid} pgid=${pgid}"
    log "cmd=${cmd}"
    _signal_target "INT" "${pid}" "${pgid}" "${own_pgid}"
    if _wait_for_stop 20 "${pid}" "${pgid}" "${own_pgid}"; then
      continue
    fi
    _signal_target "TERM" "${pid}" "${pgid}" "${own_pgid}"
    if _wait_for_stop 20 "${pid}" "${pgid}" "${own_pgid}"; then
      continue
    fi
    _signal_target "KILL" "${pid}" "${pgid}" "${own_pgid}"
    sleep 1
  done

  mapfile -t matches < <(_find_overlay_matches)
  if ((${#matches[@]} > 0)); then
    log "WARNING: Some optimize_overlay processes still present:"
    printf '%s\n' "${matches[@]}" >&2
    return 1
  fi
  log "All optimize_overlay processes stopped."
}

collect_recent_run_ids() {
  local tmp_ids
  tmp_ids="$(mktemp)"
  trap 'rm -f "${tmp_ids}"' RETURN

  find logs artifacts/reports artifacts/execution_overlay \
    -type f -newermt "${SINCE_EXPR}" 2>/dev/null \
    | rg -o '[0-9]{8}_[0-9]{6}' -N > "${tmp_ids}" || true

  find logs -maxdepth 1 -type f -name 'ultra_*.log' -newermt "${SINCE_EXPR}" 2>/dev/null \
    -exec rg -o '^run_id=([0-9]{8}_[0-9]{6})' -r '$1' {} \; >> "${tmp_ids}" || true

  sort -u "${tmp_ids}" | sed '/^$/d'
}

collect_delete_paths() {
  local -a run_ids=("$@")

  # Required report patterns (last-day only)
  find artifacts/reports -maxdepth 1 -type f \
    \( -name '*fetch_window_failed*.jsonl' -o -name 'overlay_ultra_fetch_failures_*.jsonl' \) \
    -newermt "${SINCE_EXPR}" 2>/dev/null || true
  find artifacts/reports -maxdepth 1 -type f -name 'overlay_ultra_crash_*.txt' \
    -newermt "${SINCE_EXPR}" 2>/dev/null || true

  # Temp/partial overlay outputs (last-day only)
  find artifacts/reports -maxdepth 1 -type f \
    \( -name 'overlay_ultra_partial_*.json' -o -name 'overlay_ultra_partial_*.jsonl' \) \
    -newermt "${SINCE_EXPR}" 2>/dev/null || true
  find artifacts/execution_overlay -mindepth 2 -maxdepth 2 -type d \
    -newermt "${SINCE_EXPR}" 2>/dev/null || true

  # Recent launcher logs
  find logs -maxdepth 1 -type f -name 'ultra_*.log' -newermt "${SINCE_EXPR}" 2>/dev/null || true
  if [[ -f logs/ultra.pid ]]; then
    echo "logs/ultra.pid"
  fi

  # Run/checkpoint entries matching discovered run IDs
  local rid
  for rid in "${run_ids[@]}"; do
    find artifacts -type f \( -path '*/runs/*' -o -path '*/checkpoints/*' \) \
      -name "*${rid}*" 2>/dev/null || true
    find artifacts -type d \( -path '*/runs/*' -o -path '*/checkpoints/*' \) \
      -name "*${rid}*" 2>/dev/null || true
  done
}

delete_paths() {
  local -a paths=("$@")
  local p
  local deleted=0
  for p in "${paths[@]}"; do
    [[ -n "${p}" ]] || continue
    [[ -e "${p}" ]] || continue
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      log "DRY-RUN delete: ${p}"
      continue
    fi
    rm -rf -- "${p}"
    log "Deleted: ${p}"
    deleted=$((deleted + 1))
  done
  log "Delete count=${deleted}"
}

main() {
  log "Repo root: ${REPO_ROOT}"
  log "Cutoff: ${SINCE_EXPR}"
  stop_overlay_processes

  mapfile -t run_ids < <(collect_recent_run_ids)
  if ((${#run_ids[@]} > 0)); then
    log "Discovered run_ids: ${run_ids[*]}"
  else
    log "No run_id discovered from last ${SINCE_HOURS}h files."
  fi

  mapfile -t paths < <(collect_delete_paths "${run_ids[@]}" | sort -u)
  if ((${#paths[@]} == 0)); then
    log "No matching last-day overlay artifacts to delete."
    exit 0
  fi

  log "Paths to delete: ${#paths[@]}"
  delete_paths "${paths[@]}"
  log "Cleanup complete."
}

main "$@"
