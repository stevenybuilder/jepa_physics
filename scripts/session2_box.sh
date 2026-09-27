#!/usr/bin/env bash
# GPU session 2: local driver + box-side run order. Nothing here rents, stops or destroys an instance.
# Before touching a shared box: read /Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md and record the lease
# (project world_mechanics_takehome, purpose "session 2 propagation + predictor", output root $REMOTE).
#
# Box image and pins (same as artifacts/gpu_session1.json): pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
# (python 3.10.14, torch 2.2.2 cu121), transformers 4.56.2, safetensors 0.8.0, av 17.1.0, numpy 1.26.4,
# pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3. >= 16 GB VRAM (batch 16 fits a 4080S), 40 GB disk.
#
# Usage (from the repo root on the laptop):
#   export HOST=<box ip> PORT=<ssh port>          # user root
#   bash scripts/session2_box.sh plan             # local CPU: carriers, targets, all edit deltas, twins (~3 min)
#   bash scripts/session2_box.sh push             # code + data + plan + stimuli -> box
#   bash scripts/session2_box.sh setup            # pip pins + model download + GPU check on the box
#   bash scripts/session2_box.sh run              # nohup: parity -> forward -> timerev -> extract-stimuli -> sha256
#   bash scripts/session2_box.sh status           # tail the box log
#   bash scripts/session2_box.sh pull             # rsync results back + sha256 verify
#   bash scripts/session2_box.sh score            # local CPU readouts -> results/session2_*.json
#
# Cost estimate, from the CPU smoke (2 carriers x 2 targets x 5 arms x layers 12, 22 on the Intel Mac: one full
# 16-frame forward 14.9 s; context-only encode + predictor 0.59 of that; suffix 0.007 + 0.039 per remaining block;
# edited context + predictor 0.168 + 0.011 per block), projected to 200 carriers x 4 targets x 6 arms x layers
# 2, 8, 12, 22: sources 320 + twins 1,270 + edits 15,840 + time-reversed 1,500 + stimuli 1,570 = ~20,500 full-forward
# equivalents; at 0.16 s each (gpu_session1, RTX 4080 SUPER, batch 16) ~55 GPU-minutes. With ~10 min of setup and
# model download, ~65 min billed: ~$0.32 at $0.295/h (4080S), ~$0.50 on a 4090 at ~$0.45/h. The forward stage
# rewrites artifacts/session2/forward/cost_estimate.json from the box's own timings after the first run.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${REMOTE:-/workspace/wm}"
HOST="${HOST:-}"
PORT="${PORT:-22}"
PY_LOCAL="$REPO/.venv/bin/python"
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new "root@$HOST")
RSYNC=(rsync -az --info=stats1 -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new")

need_host() { [[ -n "$HOST" ]] || { echo "set HOST and PORT"; exit 1; }; }

case "${1:-}" in
plan)
  cd "$REPO"
  "$PY_LOCAL" scripts/run_session2.py plan --n-carriers 200 --n-targets 4 --holdout contiguous --part1-basis design
  # stimuli sets (spec §6.2 follow-up); skip if already rendered
  [[ -f artifacts/stimuli/hard/manifest.jsonl ]] || "$PY_LOCAL" scripts/render_hard_stimuli.py
  ;;

push)
  need_host
  cd "$REPO"
  "${SSH[@]}" "mkdir -p $REMOTE/artifacts/session2 $REMOTE/artifacts/stimuli $REMOTE/vjepa-physics-takehome-4E00"
  "${RSYNC[@]}" --exclude '__pycache__' src scripts splits pyproject.toml requirements.txt "root@$HOST:$REMOTE/"
  "${RSYNC[@]}" vjepa-physics-takehome-4E00/data "root@$HOST:$REMOTE/vjepa-physics-takehome-4E00/"
  "${RSYNC[@]}" artifacts/session2/plan.npz artifacts/session2/plan.json artifacts/session2/twins \
      "root@$HOST:$REMOTE/artifacts/session2/"
  "${RSYNC[@]}" artifacts/stimuli/paper_layout artifacts/stimuli/hard "root@$HOST:$REMOTE/artifacts/stimuli/"
  # results/ is needed only for provenance/layer lookups in plan/score (both local); push the step-1 file anyway
  "${SSH[@]}" "mkdir -p $REMOTE/results"
  "${RSYNC[@]}" results/p1a_direction_direction_meanpool.json "root@$HOST:$REMOTE/results/"
  ;;

setup)
  need_host
  "${SSH[@]}" bash -s <<EOF
set -euo pipefail
cd $REMOTE
python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda)"
pip install -q transformers==4.56.2 safetensors==0.8.0 av==17.1.0 numpy==1.26.4 pandas==2.3.3 \
    scikit-learn==1.7.2 scipy==1.15.3 pytest
python -c "from transformers import VJEPA2Model; VJEPA2Model.from_pretrained('facebook/vjepa2-vitl-fpc64-256')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'PY'
import av, numpy, pandas, scipy, sklearn, torch, transformers
print({m.__name__: m.__version__ for m in (av, numpy, pandas, scipy, sklearn, torch, transformers)})
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
EOF
  ;;

run)
  need_host
  "${SSH[@]}" bash -s <<EOF
set -euo pipefail
cd $REMOTE
cat > run_session2_box.sh <<'RUN'
set -euo pipefail
cd $REMOTE
export PYTHONUNBUFFERED=1
date -u +"start %FT%TZ"
python scripts/run_session2.py parity                    # gate: per-layer rel < 1e-3 (hard), <= 1e-4 expected
python scripts/run_session2.py forward --group 8 --batch-size 16
python scripts/run_session2.py timerev --batch-size 16
python scripts/run_session2.py extract-stimuli --batch-size 16
cd artifacts
find session2/forward activations/direction/vjepa2_timerev activations/stimuli_paper_layout activations/stimuli_hard \
    -type f ! -name 'chunk_*' | sort | xargs sha256sum > session2_sha256_box.txt
date -u +"done %FT%TZ" | tee session2_DONE
RUN
nohup bash run_session2_box.sh > session2.log 2>&1 &
echo "launched pid \$!"
EOF
  ;;

status)
  need_host
  "${SSH[@]}" "tail -n 25 $REMOTE/session2.log; ls $REMOTE/artifacts/session2/forward 2>/dev/null | wc -l; \
               test -f $REMOTE/artifacts/session2_DONE && echo DONE || echo running; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
  ;;

pull)
  need_host
  cd "$REPO"
  "${SSH[@]}" "test -f $REMOTE/artifacts/session2_DONE" || { echo "box run not finished (no session2_DONE)"; exit 1; }
  "${RSYNC[@]}" "root@$HOST:$REMOTE/artifacts/session2/forward" artifacts/session2/
  mkdir -p artifacts/activations/direction
  "${RSYNC[@]}" --exclude 'chunk_*' "root@$HOST:$REMOTE/artifacts/activations/direction/vjepa2_timerev" artifacts/activations/direction/
  "${RSYNC[@]}" --exclude 'chunk_*' "root@$HOST:$REMOTE/artifacts/activations/stimuli_paper_layout" \
      "root@$HOST:$REMOTE/artifacts/activations/stimuli_hard" artifacts/activations/
  "${RSYNC[@]}" "root@$HOST:$REMOTE/artifacts/session2_sha256_box.txt" "root@$HOST:$REMOTE/session2.log" artifacts/
  (cd artifacts && shasum -a 256 -c session2_sha256_box.txt | tee session2_sha256_check.txt | grep -v ': OK$' || true)
  n_bad=$(grep -vc ': OK$' artifacts/session2_sha256_check.txt || true)
  echo "sha256 mismatches: $n_bad"
  ;;

score)
  cd "$REPO"
  "$PY_LOCAL" scripts/run_session2.py score
  ;;

*)
  sed -n '2,25p' "$0"
  exit 1
  ;;
esac
