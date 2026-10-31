#!/usr/bin/env bash
# Phase 3, v4 crafting test on Colab (rerun after the 2026-10-31 runtime cut-off).
#   bash experiments/colab_v4_chains.sh reach    # ~10 min: is the fast planner as good as the default (51% / 35%)?
#   bash experiments/colab_v4_chains.sh chains   # 3 chain runs in the background: v4, v4 + threat sense x1 and x3
# Safe to re-run: finished steps are skipped and chain runs pick up from <out>.worlds.jsonl.
set -euo pipefail
D=${D:-/content/drive/MyDrive/jepa-game-agent}
S=$D/phase1/shards
O=$D/phase3/v4
M=$O/wm_v4.pt
export JAX_PLATFORMS=cpu
FAST="--half --samples 128 --elites 16 --iterations 4"

case "${1:-}" in
reach)
    [ -f "$O/reach_fast.json" ] || jga plan --checkpoint "$M" --data "$D/phase1/test.npz" \
        --env Craftax-Classic-Pixels-v1 --k 5 10 --tasks 100 $FAST --out "$O/reach_fast.json" >"$O/reach_fast.log" 2>&1
    tail -n 1 "$O/reach_fast.log"
    ;;
chains)
    R="--chain --resume --goal-data $S/ppo_00.npz $S/ppo_01.npz $S/ppo_02.npz --goal-split train
       --achievement collect_wood:2 place_table collect_wood:1 make_wood_pickaxe collect_stone
       --resets 50 --budget 400 --env Craftax-Classic-Pixels-v1 --prior $O/prior_v4.pt
       --goal-as both --travel before --reach 0.5 --checkpoint $M --top $O/top_v4.pt --top-horizon 5 $FAST"
    run() {  # name, extra flags
        [ -f "$O/$1.json" ] && { echo "$1 done"; return; }
        # shellcheck disable=SC2086
        nohup jga plan-achievement $R ${2:-} --out "$O/$1.json" >>"$O/$1.log" 2>&1 &
        echo "$1 started (log $O/$1.log)"
    }
    run chain_fast
    run chain_t1 "--threat $O/threat_v4.pt --threat-weight 1"
    run chain_t3 "--threat $O/threat_v4.pt --threat-weight 3"
    ;;
*)
    echo "usage: $0 reach|chains" >&2
    exit 2
    ;;
esac
