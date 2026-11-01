#!/usr/bin/env bash
# Frame benchmark on this machine's GPU (Linux / macOS).
#   scripts/bench.sh                 # low, medium, high x lap, niva at 1920x1080
#   PRESETS="medium" SCENES="niva" WIDTH=2560 HEIGHT=1440 scripts/bench.sh
# Results: bench-results/<preset>-<scene>.json and bench-results/summary.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
PRESETS=${PRESETS:-"low medium high"}
SCENES=${SCENES:-"lap niva"}
WIDTH=${WIDTH:-1920}
HEIGHT=${HEIGHT:-1080}
FRAMES=${FRAMES:-1200}
cargo build --release
mkdir -p bench-results
summary=bench-results/summary.txt
echo "Silt bench $(date -Iseconds)  ${WIDTH}x${HEIGHT}" > "$summary"
for scene in $SCENES; do
  for preset in $PRESETS; do
    echo "== $preset / $scene =="
    ./target/release/silt bench --quality "$preset" --scene "$scene" \
      --width "$WIDTH" --height "$HEIGHT" --frames "$FRAMES" \
      --out "bench-results/$preset-$scene.json" 2>/dev/null \
      | grep -E '^(BENCH|GPU_)' | tee -a "$summary"
    echo >> "$summary"
  done
done
echo "Done: $summary"
