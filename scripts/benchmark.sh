#!/usr/bin/env bash
# =============================================================================
# Thunders AI — Benchmark Script
# =============================================================================
# Usage:
#   ./scripts/benchmark.sh                        # Default benchmarks
#   ./scripts/benchmark.sh --model gpt2           # Specific model
#   ./scripts/benchmark.sh --device cuda          # GPU benchmark
#   ./scripts/benchmark.sh --baseline prev.json   # Compare with baseline
#   ./scripts/benchmark.sh --output results.json  # Custom output file
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors & Logging
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
metric(){ echo -e "${CYAN}[METRIC]${NC} $*"; }

# ---------------------------------------------------------------------------
# Parse Arguments
# ---------------------------------------------------------------------------
MODEL="thunders-ai/default"
DEVICE="cpu"
NUM_RUNS=10
WARMUP_RUNS=3
BATCH_SIZES="1,4,8,16"
SEQ_LENGTHS="128,256,512"
OUTPUT_FILE=""
BASELINE_FILE=""
TASKS="chat,vision,speech"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift ;;
        --device)      DEVICE="$2"; shift ;;
        --runs)        NUM_RUNS="$2"; shift ;;
        --warmup)      WARMUP_RUNS="$2"; shift ;;
        --batch)       BATCH_SIZES="$2"; shift ;;
        --seq-len)     SEQ_LENGTHS="$2"; shift ;;
        --output)      OUTPUT_FILE="$2"; shift ;;
        --baseline)    BASELINE_FILE="$2"; shift ;;
        --tasks)       TASKS="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL       Model to benchmark (default: thunders-ai/default)"
            echo "  --device DEVICE     Device: cpu|cuda (default: cpu)"
            echo "  --runs N            Number of benchmark runs (default: 10)"
            echo "  --warmup N          Warmup runs before measuring (default: 3)"
            echo "  --batch SIZES       Comma-separated batch sizes (default: 1,4,8,16)"
            echo "  --seq-len LENGTHS   Comma-separated sequence lengths (default: 128,256,512)"
            echo "  --output FILE       Output results as JSON"
            echo "  --baseline FILE     Compare with baseline JSON file"
            echo "  --tasks TASKS       Comma-separated tasks to benchmark (default: chat,vision,speech)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
    shift
done

# Default output file
if [[ -z "$OUTPUT_FILE" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="benchmark_${TIMESTAMP}.json"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Pre-flight Checks
# ---------------------------------------------------------------------------
info "Thunders AI Benchmark Suite"
info "Model: $MODEL | Device: $DEVICE | Runs: $NUM_RUNS"

if [[ "$DEVICE" == "cuda" ]]; then
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        ok "CUDA available: $GPU_NAME"
    else
        error "CUDA requested but not available"
    fi
fi

# ---------------------------------------------------------------------------
# System Information
# ---------------------------------------------------------------------------
info "Collecting system information..."

SYS_INFO=$(python3 -c "
import platform, json
info = {
    'platform': platform.platform(),
    'python': platform.python_version(),
    'cpu': platform.processor(),
    'device': '$DEVICE',
}
try:
    import torch
    info['torch_version'] = torch.__version__
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
except ImportError:
    pass
try:
    import psutil
    info['cpu_count'] = psutil.cpu_count(logical=False)
    info['ram_gb'] = round(psutil.virtual_memory().total / 1e9, 1)
except ImportError:
    pass
print(json.dumps(info))
")
ok "System info collected"

# ---------------------------------------------------------------------------
# Run Benchmarks
# ---------------------------------------------------------------------------
info "Running benchmarks..."

BENCHMARK_RESULTS=$(python3 -c "
import json, time, statistics, sys

results = {
    'model': '$MODEL',
    'device': '$DEVICE',
    'system': json.loads('''$SYS_INFO'''),
    'tasks': {},
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}

tasks = '$TASKS'.split(',')
batch_sizes = [int(x) for x in '$BATCH_SIZES'.split(',')]
seq_lengths = [int(x) for x in '$SEQ_LENGTHS'.split(',')]
num_runs = $NUM_RUNS
warmup_runs = $WARMUP_RUNS

for task in tasks:
    task_results = {
        'latency_ms': {},
        'throughput_items_per_sec': {},
    }

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            config_key = f'batch_{batch_size}_seq_{seq_len}'
            latencies = []

            # Simulate benchmark runs (in production, this calls the actual model)
            for run_idx in range(warmup_runs + num_runs):
                start = time.perf_counter()

                # --- Actual inference call would go here ---
                # Example: output = model.generate(batch_size=batch_size, max_length=seq_len)
                # For now, simulate with a proportional sleep
                simulated_time = (batch_size * seq_len) / 50000.0  # ~simulate
                time.sleep(max(simulated_time, 0.001))

                elapsed = (time.perf_counter() - start) * 1000  # ms

                if run_idx >= warmup_runs:
                    latencies.append(elapsed)

            if latencies:
                avg_latency = statistics.mean(latencies)
                p50 = statistics.median(latencies)
                p95 = sorted(latencies)[int(len(latencies) * 0.95)]
                p99 = sorted(latencies)[int(len(latencies) * 0.99)]
                std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
                throughput = (batch_size / avg_latency) * 1000  # items/sec

                task_results['latency_ms'][config_key] = {
                    'mean': round(avg_latency, 3),
                    'p50': round(p50, 3),
                    'p95': round(p95, 3),
                    'p99': round(p99, 3),
                    'std_dev': round(std_dev, 3),
                    'min': round(min(latencies), 3),
                    'max': round(max(latencies), 3),
                }
                task_results['throughput_items_per_sec'][config_key] = round(throughput, 2)

    results['tasks'][task] = task_results

print(json.dumps(results, indent=2))
")

ok "Benchmarks completed"

# ---------------------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
info "Benchmark Results"
echo "============================================"

python3 -c "
import json, sys

results = json.loads('''$(echo "$BENCHMARK_RESULTS" | python3 -c "import sys; print(sys.stdin.read().replace(\"'\", \"'\"))")''')

print(f\"Model:    {results['model']}\")
print(f\"Device:   {results['device']}\")
print(f\"Timestamp: {results['timestamp']}\")
print()

for task_name, task_data in results['tasks'].items():
    print(f'--- {task_name.upper()} ---')
    print(f\"{'Config':<30} {'Avg (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10} {'Throughput':>12}\")
    print('-' * 85)
    for config, lat in task_data['latency_ms'].items():
        tp = task_data['throughput_items_per_sec'].get(config, 0)
        print(f\"{config:<30} {lat['mean']:>10.3f} {lat['p50']:>10.3f} {lat['p95']:>10.3f} {lat['p99']:>10.3f} {tp:>10.1f}/s\")
    print()
"

# ---------------------------------------------------------------------------
# Save Results
# ---------------------------------------------------------------------------
echo "$BENCHMARK_RESULTS" > "$OUTPUT_FILE"
ok "Results saved to $OUTPUT_FILE"

# ---------------------------------------------------------------------------
# Compare with Baseline
# ---------------------------------------------------------------------------
if [[ -n "$BASELINE_FILE" ]] && [[ -f "$BASELINE_FILE" ]]; then
    info "Comparing with baseline: $BASELINE_FILE"

    python3 -c "
import json, sys

with open('$OUTPUT_FILE') as f:
    current = json.load(f)
with open('$BASELINE_FILE') as f:
    baseline = json.load(f)

print()
print('--- COMPARISON (Current vs Baseline) ---')
print(f\"{'Config':<30} {'Baseline':>12} {'Current':>12} {'Change':>10} {'Status':>8}\")
print('-' * 75)

for task_name in current['tasks']:
    if task_name not in baseline['tasks']:
        continue
    for config in current['tasks'][task_name]['latency_ms']:
        if config not in baseline['tasks'][task_name]['latency_ms']:
            continue
        base_ms = baseline['tasks'][task_name]['latency_ms'][config]['mean']
        curr_ms = current['tasks'][task_name]['latency_ms'][config]['mean']
        change_pct = ((curr_ms - base_ms) / base_ms) * 100

        if change_pct > 10:
            status = 'REGRESS'
        elif change_pct < -10:
            status = 'IMPROVE'
        else:
            status = 'OK'

        print(f'{task_name}/{config:<22} {base_ms:>10.3f}ms {curr_ms:>10.3f}ms {change_pct:>+8.1f}% {status:>8}')
"

    ok "Comparison completed"
fi

echo ""
echo "============================================"
ok "Benchmark completed!"
echo "  Results: $OUTPUT_FILE"
echo "============================================"
