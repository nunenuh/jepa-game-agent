from svgkit import box, text, path, write
import sys  # usage: python docs/diagrams/how_it_works.py docs/assets
W = 910
body = "\n".join([
    # stage 1
    text(4, 14, "Step 1 · phase 1 · PPO's only job: record training games", "h"),
    box(4, 30, 260, 70, "PPO player", "learns from the game score", "200M steps on Colab A100", "good"),
    box(320, 30, 250, 70, "Plays Craftax", "wood, table, pickaxe, iron ...", ""),
    box(626, 30, 274, 70, "Recorded games", "~1M frames + moves", "no score kept"),
    path("M264,65 H318"), path("M570,65 H624"),
    text(4, 122, "Like hiring a skilled gamer to record gameplay videos. After this, PPO is not used again.", "c"),
    path("M763,102 V168"), text(772, 140, "only frames + moves go on", "c"),
    # stage 2
    text(4, 162, "Step 2 · phase 2 · the JEPA agent (the real project)", "h"),
    box(4, 178, 380, 70, "JEPA world model learns", "how the game reacts to each move", "from the recordings · no score, no PPO", "learn"),
    box(626, 178, 274, 70, "Recorded games", '"I pressed this, then saw that"', ""),
    path("M624,213 H386"),
    box(4, 270, 260, 70, "Goal picture", '"have a pickaxe"', "", "frozen"),
    box(320, 270, 250, 70, "JEPA imagines", "futures of candidate moves", "inside its own model", "learn"),
    box(626, 270, 274, 70, "Picks the best moves", "and plays them in the game", "planning, not RL", "learn"),
    path("M264,305 H318"), path("M570,305 H624"), path("M360,250 V268"),
    path("M445,342 V408"), text(454, 380, "only if planning is too slow (~1 s per move)", "c"),
    # stage 3
    text(4, 402, "Step 3 · phase 5 · optional speed-up", "h"),
    box(4, 418, 260, 76, "JEPA's imagination", "a fake game inside", "the world model", "learn"),
    box(320, 418, 250, 76, "Small fast policy", "copies the planner, or RL", "(PPO / Dreamer) inside imagination", "frozen"),
    box(626, 418, 274, 76, "Real-time play", "instant moves", ""),
    path("M264,456 H318"), path("M570,456 H624"),
    # legend
    text(4, 528, "PPO made the training data  ·  JEPA understands the game and plans  ·  RL again only later, only for speed", "c"),
])
write(sys.argv[1], "how-it-works", W, 545, body)
