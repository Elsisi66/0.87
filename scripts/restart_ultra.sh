#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
PID_FILE="${LOG_DIR}/ultra.pid"

ACTION="${1:-restart}"

# ULTRA defaults (override via env if needed).
EVAL_PROCS="${EVAL_PROCS:-3}"
FETCH_WORKERS="${FETCH_WORKERS:-1}"
EARLY_STOP="${EARLY_STOP:-30}"
MAX_CACHE_MB="${MAX_CACHE_MB:-768}"
MAX_RSS_MB="${MAX_RSS_MB:-6000}"

SYMBOLS="${SYMBOLS:-BTCUSDT,ADAUSDT,AVAXUSDT,SOLUSDT}"
START_DATE="${START_DATE:-2017-01-01}"
END_DATE="${END_DATE:-2025-12-31}"
TEST_START="${TEST_START:-2024-01-01}"
TEST_END="${TEST_END:-2025-12-31}"
INITIAL_EQUITY="${INITIAL_EQUITY:-100}"
MC_SPLITS="${MC_SPLITS:-30}"
SPLIT_SEEDS="${SPLIT_SEEDS:-101,202,303}"
RANDOM_SAMPLES="${RANDOM_SAMPLES:-2000}"
GA_GENERATIONS="${GA_GENERATIONS:-250}"
POP_SIZE="${POP_SIZE:-80}"

# Thread caps to avoid BLAS/OpenMP oversubscription inside multiprocessing workloads.
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

TARGET_PID=""
TARGET_PGID=""
TARGET_PPID=""
TARGET_CMD=""

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/restart_ultra.sh stop
  scripts/restart_ultra.sh start
  scripts/restart_ultra.sh restart

Defaults can be overridden with environment variables:
  EVAL_PROCS FETCH_WORKERS EARLY_STOP MAX_CACHE_MB MAX_RSS_MB
  OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
EOF
}

find_ultra_matches() {
  ps -eo pid=,pgid=,ppid=,args= | awk -v self_pid="$$" '
    {
      pid=$1; pgid=$2; ppid=$3;
      $1=""; $2=""; $3="";
      sub(/^[[:space:]]+/, "", $0);
      has_target = ($0 ~ /(^|[[:space:]\/])optimize_overlay\.py([[:space:]]|$)/) || ($0 ~ /(^|[[:space:]])-m[[:space:]]+bot087\.cli\.optimize_overlay([[:space:]]|$)/);
      has_ultra = ($0 ~ /(^|[[:space:]])--search([=[:space:]]+)ultra([[:space:]]|$)/);
      if (pid != self_pid && has_target && has_ultra) {
        printf "%s|%s|%s|%s\n", pid, pgid, ppid, $0;
      }
    }
  '
}

print_match_diagnostics() {
  local line pid pgid ppid cmd
  local pids=()
  log "Matched process diagnostics:"
  for line in "$@"; do
    IFS='|' read -r pid pgid ppid cmd <<< "${line}"
    pids+=("${pid}")
    log "  pid=${pid} pgid=${pgid} ppid=${ppid} cmd=${cmd}"
  done
  if ((${#pids[@]} > 0)); then
    local pid_csv
    pid_csv="$(IFS=,; echo "${pids[*]}")"
    ps -fp "${pid_csv}" || true
  fi
}

resolve_target() {
  local matches line pid pgid ppid cmd
  local -a all_matches=()
  local -a root_matches=()
  local -A by_pid=()

  mapfile -t all_matches < <(find_ultra_matches)
  if ((${#all_matches[@]} == 0)); then
    return 1
  fi

  for line in "${all_matches[@]}"; do
    IFS='|' read -r pid pgid ppid cmd <<< "${line}"
    by_pid["${pid}"]=1
  done

  for line in "${all_matches[@]}"; do
    IFS='|' read -r pid pgid ppid cmd <<< "${line}"
    if [[ -z "${by_pid[${ppid}]:-}" ]]; then
      root_matches+=("${line}")
    fi
  done

  if ((${#root_matches[@]} != 1)); then
    log "ERROR: Ambiguous optimize_overlay ultra matches (total=${#all_matches[@]} roots=${#root_matches[@]})."
    print_match_diagnostics "${all_matches[@]}"
    return 2
  fi

  IFS='|' read -r TARGET_PID TARGET_PGID TARGET_PPID TARGET_CMD <<< "${root_matches[0]}"
  return 0
}

self_pgid() {
  ps -o pgid= -p "$$" | tr -d ' '
}

list_related_processes() {
  local pid="$1"
  local pgid="$2"
  local own_pgid="$3"

  if [[ -z "${pid}" ]]; then
    return 0
  fi

  if [[ -n "${pgid}" && "${pgid}" != "${own_pgid}" ]]; then
    ps -eo pid=,pgid=,ppid=,etime=,stat=,args= | awk -v p="${pid}" -v g="${pgid}" '
      ($1 == p) || ($2 == g) || ($3 == p) { print }
    '
  else
    ps -eo pid=,pgid=,ppid=,etime=,stat=,args= | awk -v p="${pid}" '
      ($1 == p) || ($3 == p) { print }
    '
  fi
}

job_alive() {
  local pid="$1"
  local pgid="$2"
  local own_pgid="$3"

  if kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi

  if [[ -n "${pgid}" && "${pgid}" != "${own_pgid}" ]]; then
    if ps -eo pgid= | awk -v g="${pgid}" '$1 == g { found=1; exit 0 } END { exit (found ? 0 : 1) }'; then
      return 0
    fi
  fi

  return 1
}

signal_target() {
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

wait_for_stop() {
  local timeout_sec="$1"
  local stage_name="$2"
  local pid="$3"
  local pgid="$4"
  local own_pgid="$5"
  local deadline=$((SECONDS + timeout_sec))

  while ((SECONDS < deadline)); do
    if ! job_alive "${pid}" "${pgid}" "${own_pgid}"; then
      return 0
    fi
    sleep 1
  done

  log "${stage_name} wait timed out after ${timeout_sec}s. Remaining related processes:"
  list_related_processes "${pid}" "${pgid}" "${own_pgid}" || true
  return 1
}

stop_run() {
  local rc own_pgid
  set +e
  resolve_target
  rc=$?
  set -e

  if [[ "${rc}" -eq 1 ]]; then
    log "No running optimize_overlay ultra process found."
    rm -f "${PID_FILE}"
    return 0
  fi
  if [[ "${rc}" -eq 2 ]]; then
    return 2
  fi

  own_pgid="$(self_pgid)"
  log "Stopping target pid=${TARGET_PID} pgid=${TARGET_PGID} ppid=${TARGET_PPID}"
  log "Target cmd: ${TARGET_CMD}"
  log "Related processes before stop:"
  list_related_processes "${TARGET_PID}" "${TARGET_PGID}" "${own_pgid}" || true

  log "Sending SIGINT..."
  signal_target "INT" "${TARGET_PID}" "${TARGET_PGID}" "${own_pgid}"
  if wait_for_stop 60 "SIGINT" "${TARGET_PID}" "${TARGET_PGID}" "${own_pgid}"; then
    log "Process exited cleanly after SIGINT."
    rm -f "${PID_FILE}"
    return 0
  fi

  log "Sending SIGTERM..."
  signal_target "TERM" "${TARGET_PID}" "${TARGET_PGID}" "${own_pgid}"
  if wait_for_stop 30 "SIGTERM" "${TARGET_PID}" "${TARGET_PGID}" "${own_pgid}"; then
    log "Process exited after SIGTERM."
    rm -f "${PID_FILE}"
    return 0
  fi

  log "Sending SIGKILL..."
  signal_target "KILL" "${TARGET_PID}" "${TARGET_PGID}" "${own_pgid}"
  sleep 2
  if job_alive "${TARGET_PID}" "${TARGET_PGID}" "${own_pgid}"; then
    log "ERROR: process still alive after SIGKILL. Remaining related processes:"
    list_related_processes "${TARGET_PID}" "${TARGET_PGID}" "${own_pgid}" || true
    return 1
  fi

  log "Process exited after SIGKILL."
  rm -f "${PID_FILE}"
  return 0
}

python_bin() {
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    printf '%s\n' "${REPO_ROOT}/.venv/bin/python"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  return 1
}

start_run() {
  local rc py ts log_file new_pid
  local -a cmd

  set +e
  resolve_target
  rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    log "ERROR: Existing optimize_overlay ultra process is still running (pid=${TARGET_PID}); stop it before start."
    return 1
  fi
  if [[ "${rc}" -eq 2 ]]; then
    return 2
  fi

  py="$(python_bin)" || {
    log "ERROR: Could not find a Python interpreter."
    return 1
  }

  mkdir -p "${LOG_DIR}"
  ts="$(date -u +%Y%m%d_%H%M%S)"
  log_file="${LOG_DIR}/ultra_${ts}.log"

  cmd=(
    "${py}" -u -m bot087.cli.optimize_overlay
    --search ultra
    --symbols "${SYMBOLS}"
    --symbols-mode sequential
    --eval-procs "${EVAL_PROCS}"
    --fetch-workers "${FETCH_WORKERS}"
    --cache-1s true
    --max-cache-mb "${MAX_CACHE_MB}"
    --max-rss-mb "${MAX_RSS_MB}"
    --start "${START_DATE}"
    --end "${END_DATE}"
    --initial-equity "${INITIAL_EQUITY}"
    --mc-splits "${MC_SPLITS}"
    --split-seeds "${SPLIT_SEEDS}"
    --random-samples "${RANDOM_SAMPLES}"
    --ga-generations "${GA_GENERATIONS}"
    --pop-size "${POP_SIZE}"
    --early-stop "${EARLY_STOP}"
    --test-start "${TEST_START}"
    --test-end "${TEST_END}"
    --save-active true
  )

  {
    echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo_root=${REPO_ROOT}"
    echo "thread_caps=OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"
    printf 'command='
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "${log_file}"

  (
    cd "${REPO_ROOT}"
    if command -v setsid >/dev/null 2>&1; then
      nohup setsid env \
        OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
        OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
        NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
        "${cmd[@]}" </dev/null >> "${log_file}" 2>&1 &
    else
      nohup env \
        OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
        OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
        NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
        "${cmd[@]}" </dev/null >> "${log_file}" 2>&1 &
    fi
    new_pid=$!
    disown "${new_pid}" 2>/dev/null || true
    echo "${new_pid}" > "${PID_FILE}.tmp"
  )

  new_pid="$(cat "${PID_FILE}.tmp")"
  rm -f "${PID_FILE}.tmp"
  echo "${new_pid}" > "${PID_FILE}"
  sleep 1

  if ! kill -0 "${new_pid}" 2>/dev/null; then
    log "ERROR: Started process pid=${new_pid} is not running. Check log: ${log_file}"
    return 1
  fi

  log "Started ULTRA run pid=${new_pid}"
  log "PID file: ${PID_FILE}"
  log "Log file: ${log_file}"
  return 0
}

main() {
  case "${ACTION}" in
    stop)
      stop_run
      ;;
    start)
      start_run
      ;;
    restart)
      stop_run
      start_run
      ;;
    *)
      usage
      return 1
      ;;
  esac
}

main
