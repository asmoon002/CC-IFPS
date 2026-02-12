
# Implementation Plan

## [Goal]
Synthesize the "Golden Configuration" script to validate the hybrid PatchCore model (Greedy + D2) and achieve the target AP of ~0.6100+.

## [Proposed Strategy]
Create `run_final_best_3runs.sh` which:
1. Iterates through all 15 MVTec AD classes.
2. Uses a `case` statement to assign the winner sampler (`greedy` vs `d2`) and optimal hyperparameters for each class.
3. Runs the experiment 3 times (seeds 0, 1, 2) to get mean/std.

## [Configuration Table]
| Class | Sampler | Layers | Tau | k | Budget |
|---|---|---|---|---|---|
| bottle | greedy | 2-3 | 0.01 | 1 | 70k |
| cable | greedy | 2-3 | 0.02 | 1 | 70k |
| capsule | greedy | 2-3 | 0.03 | 1 | 50k |
| carpet | d2 | 1-2-3 | 0.03 | 1 | 50k |
| grid | d2 | 1-2-3 | 0.02 | 1 | 70k |
| hazelnut | greedy | 1-2-3 | 0.02 | 1 | 70k |
| leather | greedy | 1-2-3 | 0.02 | 1 | 50k |
| metal_nut | greedy | 1-2-3 | 0.03 | 1 | 70k |
| pill | greedy | 1-2-3 | 0.01 | 1 | 30k |
| screw | d2 | 1-2-3 | 0.01 | 1 | 70k |
| tile | greedy | 1-2-3 | 0.02 | 1 | 70k |
| toothbrush | d2 | 2-3 | 0.01 | 1 | 50k |
| transistor | greedy | 2-3 | 0.02 | 1 | 50k |
| wood | greedy | 1-2-3 | 0.01 | 1 | 30k |
| zipper | greedy | 1-2-3 | 0.01 | 1 | 70k |

## [Verification]
Run the script and check if the average AP matches or exceeds the 0.6100 benchmark.
