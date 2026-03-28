#!/bin/bash
# Autonomous experiment loop — runs cycles until interrupted
set -e
cd "$(dirname "$0")"
COMMIT=$(git rev-parse --short HEAD)

for i in $(seq 1 20); do
    echo "=== CYCLE $i ==="
    python3 harness_ane.py > run_ane.log 2>&1

    # Extract results
    VAL=$(grep '^val_loss:' run_ane.log | awk '{print $2}')
    UTIL=$(grep '^ane_util_pct:' run_ane.log | awk '{print $2}')
    STEPS=$(grep '^steps:' run_ane.log | awk '{print $2}')
    MS=$(grep '^ms_per_step:' run_ane.log | awk '{print $2}')
    STATUS=$(grep '^status:' run_ane.log | awk '{print $2}')

    if [ "$STATUS" = "crash" ] || [ -z "$VAL" ]; then
        echo "CRASH at cycle $i"
        echo -e "${COMMIT}\t0.000000\t0.0\tcrash\tACCUM=4 cycle$i CRASH" >> results.tsv
        break
    fi

    echo "cycle$i: val_loss=$VAL ane_util=$UTIL steps=$STEPS ms/step=$MS"
    echo -e "${COMMIT}\t${VAL}\t${UTIL}\tkeep\tACCUM=4 cycle$i (${STEPS} steps, ${MS}ms/step)" >> results.tsv
done

echo "=== DONE ==="
cat results.tsv
