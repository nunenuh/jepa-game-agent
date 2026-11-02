#!/usr/bin/env bash
# Phase 3, v4 crafting test on Colab (rerun after the 2026-10-31 runtime cut-off).
#   bash experiments/colab_v4_chains.sh reach    # ~10 min: is the fast planner as good as the default (51% / 35%)?
#   bash experiments/colab_v4_chains.sh chains   # v4 and v2 at once, background: the first fair comparison since the
#                                                # unlock flag bug (experiments/results), so v2 is the baseline again
#   bash experiments/colab_v4_chains.sh threat   # v4 + threat sense x1 and x3, background (after chains: costs units)
#   bash experiments/colab_v4_chains.sh memory   # does the tile map work on v4? (docs/DESIGN_MEMORY.md)
#   bash experiments/colab_v4_chains.sh progress # does v4 read item counts? then the chain on its own sense (DESIGN_RECIPE)
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
chains | threat | own)
    common="--chain --resume --goal-data $S/ppo_00.npz $S/ppo_01.npz $S/ppo_02.npz --goal-split train
       --achievement collect_wood:2 place_table collect_wood:1 make_wood_pickaxe collect_stone
       --resets 50 --budget 400 --env Craftax-Classic-Pixels-v1 --goal-as both --travel before --reach 0.5
       --top-horizon 5 $FAST --record 3"  # first 3 worlds saved to <out>.play.npz: jga render makes GIFs
    V4="--checkpoint $M --top $O/top_v4.pt --prior $O/prior_v4.pt"
    V2="--checkpoint $D/phase2/wm_inv.pt --top $D/phase3/top_s0.pt --prior $D/phase3/prior/prior_v2.pt"
    run() {  # name, flags
        [ -f "$O/$1.json" ] && { echo "$1 done"; return; }
        # shellcheck disable=SC2086
        nohup jga plan-achievement $common $2 --out "$O/$1.json" >>"$O/$1.log" 2>&1 &
        echo "$1 started (log $O/$1.log)"
    }
    if [ "$1" = chains ]; then
        run chain_v4 "$V4"
        run chain_v2 "$V2"
    elif [ "$1" = own ]; then  # after `progress`: same as chain_v4, but the agent decides "stage done" itself
        run chain_own "$V4 --progress $O/progress_v4.pt"
    else
        run chain_t1 "$V4 --threat $O/threat_v4.pt --threat-weight 1"
        run chain_t3 "$V4 --threat $O/threat_v4.pt --threat-weight 3"
    fi
    ;;
progress)
    jga train-progress --data $S/ppo_00.npz $S/ppo_01.npz $S/ppo_02.npz --split train --bottom "$M" \
        --env Craftax-Classic-Pixels-v1 --checkpoint "$O/progress_v4.pt" >>"$O/progress.log" 2>&1
    tail -n 1 "$O/progress.log"
    ;;
memory)
    for a in collect_wood make_wood_pickaxe; do  # trees; the table
        for sim in 0.8 0.9 0.95; do
            [ -f "$O/memory_${a}_$sim.json" ] || jga memory-check --checkpoint "$M" --data "$D/phase1/test.npz" \
                --env Craftax-Classic-Pixels-v1 --episodes 50 --goal-data $S/ppo_00.npz --goal-split train \
                --achievement $a --memory-sim $sim --out "$O/memory_${a}_$sim.json" >>"$O/memory.log" 2>&1
            tail -n 1 "$O/memory.log"
        done
    done
    ;;
*)
    echo "usage: $0 reach|chains|threat|own|progress|memory" >&2
    exit 2
    ;;
esac
