#!/bin/bash
# Quick sweep status checker
# Usage: bash scripts/sweep_status.sh [sweep_dir]

SWEEP_DIR="${1:-results/sweep}"

echo "=== SWEEP STATUS ==="
echo "Directory: $SWEEP_DIR"
echo ""

# Check if sweep has started
if [ ! -f "$SWEEP_DIR/sweep_plan.json" ]; then
    echo "⏳ Sweep not started yet (no sweep_plan.json found)"
    echo ""
    echo "Waiting for first run to begin..."
    echo "Check if the sweep is running: ps aux | grep ch04_sweep"
    exit 0
fi

# Count completed
TOTAL_CONFIGS=$(jq '.configs | length' "$SWEEP_DIR/sweep_plan.json" 2>/dev/null)
SEEDS=$(jq '.seeds | length' "$SWEEP_DIR/sweep_plan.json" 2>/dev/null)
SCRIPT_MODE=$(jq -r '.script_mode // "old"' "$SWEEP_DIR/sweep_plan.json" 2>/dev/null)

if [ -z "$TOTAL_CONFIGS" ] || [ -z "$SEEDS" ]; then
    echo "⚠️  Could not parse sweep_plan.json"
    exit 1
fi

EXPECTED=$((TOTAL_CONFIGS * SEEDS))
COMPLETED=$(ls "$SWEEP_DIR/results/"*_eval.json 2>/dev/null | wc -l)

echo "Script mode: $SCRIPT_MODE"
echo "Progress: $COMPLETED / $EXPECTED runs completed"

# Calculate ETA
if [ "$COMPLETED" -gt 0 ]; then
    # Get time of first and last completion
    FIRST_FILE=$(ls -t "$SWEEP_DIR/results/"*_eval.json 2>/dev/null | tail -1)
    LAST_FILE=$(ls -t "$SWEEP_DIR/results/"*_eval.json 2>/dev/null | head -1)

    if [ -f "$FIRST_FILE" ] && [ -f "$LAST_FILE" ]; then
        FIRST_TIME=$(stat -c %Y "$FIRST_FILE" 2>/dev/null || stat -f %m "$FIRST_FILE" 2>/dev/null)
        LAST_TIME=$(stat -c %Y "$LAST_FILE" 2>/dev/null || stat -f %m "$LAST_FILE" 2>/dev/null)

        if [ "$COMPLETED" -gt 1 ] && [ -n "$FIRST_TIME" ] && [ -n "$LAST_TIME" ]; then
            ELAPSED=$((LAST_TIME - FIRST_TIME))
            if [ "$ELAPSED" -gt 0 ]; then
                RATE=$(echo "scale=2; $COMPLETED / $ELAPSED * 3600" | bc 2>/dev/null)
                REMAINING=$((EXPECTED - COMPLETED))
                if [ -n "$RATE" ] && [ "$RATE" != "0" ]; then
                    ETA_HOURS=$(echo "scale=1; $REMAINING / $RATE" | bc 2>/dev/null)
                    echo "Rate: ~${RATE} runs/hour"
                    echo "ETA: ~${ETA_HOURS} hours remaining"
                fi
            fi
        fi
    fi
fi
echo ""

# Show recent completions
echo "=== RECENT COMPLETIONS ==="
if [ "$COMPLETED" -eq 0 ]; then
    echo "  (none yet - first run in progress)"
else
    ls -t "$SWEEP_DIR/results/"*_eval.json 2>/dev/null | head -5 | while read FILE; do
        SR=$(jq -r '.aggregate.success_rate' "$FILE" 2>/dev/null)
        NAME=$(basename "$FILE" _eval.json | sed 's/sweep_//')
        if [ -n "$SR" ]; then
            printf "  %5.1f%% %s\n" "$(echo "$SR * 100" | bc)" "$NAME"
        fi
    done
fi
echo ""

# Quick stats if we have results
if [ "$COMPLETED" -gt 2 ]; then
    echo "=== TOP SUCCESS RATES ==="
    for f in "$SWEEP_DIR/results/"*_eval.json; do
        SR=$(jq -r '.aggregate.success_rate' "$f" 2>/dev/null)
        NAME=$(basename "$f" _eval.json | sed 's/sweep_//')
        if [ -n "$SR" ]; then
            echo "$SR $NAME"
        fi
    done 2>/dev/null | sort -rn | head -5 | while read SR NAME; do
        printf "  %5.1f%% %s\n" "$(echo "$SR * 100" | bc)" "$NAME"
    done

    echo ""
    echo "=== BOTTOM SUCCESS RATES ==="
    for f in "$SWEEP_DIR/results/"*_eval.json; do
        SR=$(jq -r '.aggregate.success_rate' "$f" 2>/dev/null)
        NAME=$(basename "$f" _eval.json | sed 's/sweep_//')
        if [ -n "$SR" ]; then
            echo "$SR $NAME"
        fi
    done 2>/dev/null | sort -n | head -3 | while read SR NAME; do
        printf "  %5.1f%% %s\n" "$(echo "$SR * 100" | bc)" "$NAME"
    done
fi
