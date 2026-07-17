#!/usr/bin/env bash

set -uo pipefail

ROOT=${ROOT:-/home/xwj/workspace/MoE-Ascend-Opt}
MODEL=${MODEL:-/mnt/models/MiniMax-M2.5-AWQ-4bit}
DATASET=${DATASET:-/home/xwj/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json}
PORT=${PORT:-31050}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/logs/minimax_m2_5_offload_mem09_seed1_${RUN_ID}}

MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.9}
NUM_PROMPTS=${NUM_PROMPTS:-32}
OUTPUT_LEN=${OUTPUT_LEN:-320}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-32}
CACHE_PRIME_PROMPTS=${CACHE_PRIME_PROMPTS:-20}
SEED=${SEED:-1}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-1200}

CONCURRENCIES=(1 4 8)
CACHE_SIZES=(512 2048 4096)
declare -A LAYER_COUNTS=(
  [512]=2
  [2048]=8
  [4096]=16
)

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export NANOVLLM_TP_SIZE=2
export NANOVLLM_TP_THREADS_PER_NODE=${NANOVLLM_TP_THREADS_PER_NODE:-20}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-20}
unset MOE_ASCEND_NPU_DISABLE || true

mkdir -p "${RESULT_ROOT}" "${ROOT}/logs"
exec > >(tee -a "${RESULT_ROOT}/orchestrator.log") 2>&1

SERVER_PID=
CURRENT_MODE=
STATUS_FILE=${RESULT_ROOT}/status.tsv
printf 'mode\tstage\tstatus\ttime\n' >"${STATUS_FILE}"
printf '%s\n' "${RESULT_ROOT}" >"${ROOT}/logs/minimax_offload_matrix_latest_path.txt"

record_status() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date --iso-8601=seconds)" \
    >>"${STATUS_FILE}"
}

cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Stopping ${CURRENT_MODE:-server} process group ${SERVER_PID}"
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
  SERVER_PID=

  local port_pids
  port_pids=$(lsof -t -i:"${PORT}" 2>/dev/null || true)
  if [[ -n "${port_pids}" ]]; then
    kill -TERM ${port_pids} 2>/dev/null || true
    sleep 2
    port_pids=$(lsof -t -i:"${PORT}" 2>/dev/null || true)
    [[ -z "${port_pids}" ]] || kill -KILL ${port_pids} 2>/dev/null || true
  fi
}

cleanup() {
  cleanup_server
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local result_dir=$1
  local deadline=$((SECONDS + SERVER_TIMEOUT))
  echo "Waiting for ${CURRENT_MODE} on port ${PORT} (timeout=${SERVER_TIMEOUT}s)"
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "${CURRENT_MODE} is ready"
      return 0
    fi
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "${CURRENT_MODE} exited before readiness"
      tail -120 "${result_dir}/server.log" || true
      return 1
    fi
    sleep 5
  done
  echo "${CURRENT_MODE} readiness timeout"
  tail -120 "${result_dir}/server.log" || true
  return 1
}

start_server() {
  local mode=$1
  local value=$2
  local result_dir=$3
  local -a args=(
    sglang serve
    --host 127.0.0.1 --port "${PORT}"
    --model-path "${MODEL}" --trust-remote-code
    --tp-size 1 --attention-backend ascend
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --cuda-graph-bs 1 2 4 8
  )

  case "${mode}" in
    cpu)
      args+=(
        --enable-moe-offload
        --moe-offload-start-layer 0
        --moe-offload-quant-type q4_0
      )
      ;;
    layer)
      args+=(
        --enable-moe-offload
        --moe-offload-start-layer "${value}"
        --moe-offload-quant-type q4_0
      )
      ;;
    cache)
      args+=(
        --enable-moe-expert-cache
        --moe-expert-cache-size "${value}"
        --moe-expert-cache-swap-per-update 64
        --moe-expert-cache-update-interval 16
        --moe-expert-cache-warmup-steps 16
        --moe-expert-cache-decay 0.95
      )
      ;;
    *)
      echo "Unknown mode: ${mode}"
      return 2
      ;;
  esac

  printf '%q ' "${args[@]}" >"${result_dir}/server_command.txt"
  printf '\n' >>"${result_dir}/server_command.txt"
  setsid "${args[@]}" >"${result_dir}/server.log" 2>&1 &
  SERVER_PID=$!
  echo "Started ${CURRENT_MODE}: pid=${SERVER_PID}"
}

run_benchmark() {
  local result_dir=$1
  local concurrency=$2
  echo "Benchmark ${CURRENT_MODE}: concurrency=${concurrency}"
  python -m sglang.bench_serving \
    --backend sglang \
    --base-url "http://127.0.0.1:${PORT}" \
    --dataset-name sharegpt \
    --dataset-path "${DATASET}" \
    --num-prompts "${NUM_PROMPTS}" \
    --sharegpt-output-len "${OUTPUT_LEN}" \
    --max-concurrency "${concurrency}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --seed "${SEED}" \
    --output-file "${result_dir}/sharegpt_c${concurrency}.jsonl" \
    2>&1 | tee "${result_dir}/bench_c${concurrency}.log"
}

prime_cache() {
  local result_dir=$1
  echo "Untimed decode priming for ${CURRENT_MODE}: prompts=${CACHE_PRIME_PROMPTS}"
  python -m sglang.bench_serving \
    --backend sglang \
    --base-url "http://127.0.0.1:${PORT}" \
    --dataset-name sharegpt \
    --dataset-path "${DATASET}" \
    --num-prompts "${CACHE_PRIME_PROMPTS}" \
    --sharegpt-output-len "${OUTPUT_LEN}" \
    --max-concurrency 1 \
    --warmup-requests 0 \
    --seed "${SEED}" \
    --output-file "${result_dir}/cache_prime_c1.jsonl" \
    2>&1 | tee "${result_dir}/cache_prime_c1.log"
}

run_mode() {
  local label=$1
  local mode=$2
  local value=$3
  local result_dir=${RESULT_ROOT}/${label}
  CURRENT_MODE=${label}
  mkdir -p "${result_dir}"

  echo
  echo "============================================================"
  echo "Starting ${label}: mode=${mode} value=${value}"
  echo "============================================================"
  record_status "${label}" server starting
  cleanup_server
  start_server "${mode}" "${value}" "${result_dir}" || {
    record_status "${label}" server start_failed
    return 1
  }
  if ! wait_for_server "${result_dir}"; then
    record_status "${label}" server failed
    cleanup_server
    return 1
  fi
  record_status "${label}" server ready

  curl -fsS "http://127.0.0.1:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"Reply with exactly: OK\",\"max_tokens\":4,\"temperature\":0}" \
    >"${result_dir}/smoke_response.json" || true

  if [[ "${mode}" == cache ]]; then
    if prime_cache "${result_dir}"; then
      record_status "${label}" prime ok
    else
      record_status "${label}" prime failed
    fi
  fi

  local concurrency
  for concurrency in "${CONCURRENCIES[@]}"; do
    if run_benchmark "${result_dir}" "${concurrency}"; then
      record_status "${label}" "benchmark_c${concurrency}" ok
    else
      record_status "${label}" "benchmark_c${concurrency}" failed
    fi
  done

  cleanup_server
  record_status "${label}" server stopped
}

write_metadata() {
  {
    echo "run_id=${RUN_ID}"
    echo "root=${ROOT}"
    echo "model=${MODEL}"
    echo "dataset=${DATASET}"
    echo "git_commit=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || true)"
    echo "mem_fraction_static=${MEM_FRACTION_STATIC}"
    echo "npu_device=${ASCEND_RT_VISIBLE_DEVICES}"
    echo "sglang_tp_size=1"
    echo "cpu_tp_size=${NANOVLLM_TP_SIZE}"
    echo "cpu_threads_per_node=${NANOVLLM_TP_THREADS_PER_NODE}"
    echo "num_prompts=${NUM_PROMPTS}"
    echo "output_len=${OUTPUT_LEN}"
    echo "warmup_requests=${WARMUP_REQUESTS}"
    echo "cache_prime_prompts=${CACHE_PRIME_PROMPTS}"
    echo "seed=${SEED}"
    echo "concurrencies=${CONCURRENCIES[*]}"
    echo "cache_sizes=${CACHE_SIZES[*]}"
    echo "layer_counts=2 8 16"
    python - <<'PY'
import platform
print(f"python={platform.python_version()}")
try:
    import torch
    print(f"torch={torch.__version__}")
except Exception as exc:
    print(f"torch=unavailable:{exc}")
try:
    import torch_npu
    print(f"torch_npu={torch_npu.__version__}")
except Exception as exc:
    print(f"torch_npu=unavailable:{exc}")
PY
  } >"${RESULT_ROOT}/metadata.txt"
}

build_summary() {
  python - "${RESULT_ROOT}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for result_file in sorted(root.glob("*/sharegpt_c*.jsonl")):
    try:
        payload = json.loads(result_file.read_text().splitlines()[-1])
    except Exception as exc:
        rows.append({"mode": result_file.parent.name, "file": str(result_file), "error": str(exc)})
        continue
    rows.append({
        "mode": result_file.parent.name,
        "concurrency": payload.get("max_concurrency"),
        "request_throughput": payload.get("request_throughput"),
        "input_throughput": payload.get("input_throughput"),
        "output_throughput": payload.get("output_throughput"),
        "mean_ttft_ms": payload.get("mean_ttft_ms"),
        "median_ttft_ms": payload.get("median_ttft_ms"),
        "p99_ttft_ms": payload.get("p99_ttft_ms"),
        "mean_tpot_ms": payload.get("mean_tpot_ms"),
        "median_tpot_ms": payload.get("median_tpot_ms"),
        "p99_tpot_ms": payload.get("p99_tpot_ms"),
        "mean_e2e_latency_ms": payload.get("mean_e2e_latency_ms"),
        "median_e2e_latency_ms": payload.get("median_e2e_latency_ms"),
        "p99_e2e_latency_ms": payload.get("p99_e2e_latency_ms"),
        "successful_requests": payload.get("completed"),
        "file": str(result_file),
    })

(root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
fieldnames = sorted({key for row in rows for key in row})
with (root / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
}

main() {
  echo "MiniMax M2.5 expert-offload matrix started at $(date --iso-8601=seconds)"
  echo "Result root: ${RESULT_ROOT}"
  write_metadata

  run_mode cpu_q4 cpu 0 || true

  local cache_size layer_count
  for cache_size in "${CACHE_SIZES[@]}"; do
    layer_count=${LAYER_COUNTS[${cache_size}]}
    run_mode "layer_l${layer_count}" layer "${layer_count}" || true
  done

  for cache_size in "${CACHE_SIZES[@]}"; do
    run_mode "cache_k${cache_size}" cache "${cache_size}" || true
  done

  build_summary || true
  echo "MiniMax M2.5 expert-offload matrix finished at $(date --iso-8601=seconds)"
  echo "Result root: ${RESULT_ROOT}"
}

main "$@"
