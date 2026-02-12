# Task List

- [x] Analyze Thesis Documents (`chap3.tex`, `chap4.tex`) to confirm "Proposed Method" definition.
- [x] Verify `run_patchcore.py` pipeline (Preprocessing -> Feature Extraction -> Sampler -> Anomaly Scoring).
- [x] Implement Class-Adaptive D2 Sampling in `sampler.py`.
- [x] 🔄 Hybrid Optimization Strategy (The "Grand Tournament")
    - [x] Execute Comprehensive Grid Search for **D2 Sampling** (All Classes, All Params).
    - [x] Execute Comprehensive Grid Search for **Greedy Sampling** (All Classes, All Params).
    - [x] Compare results class-by-class to determine the winner (Greedy vs. D2).
- [x] **Final Synthesis & Validation**
    - [x] Create `run_final_best_3runs.sh` with the "Golden Configuration" (Best Sampler per Class).
    - [x] Execute `run_final_best_3runs.sh` (with Sigma=2) to get final paper-ready stats.
    - [x] Run `analyze_final_results.py` to generate Mean ± Std table. -> **DONE (AP: 0.6105)**
- [ ] Future Work (Post-Paper)
    - [x] Implement Density-Aware Scoring (Sampling Weights Reuse). -> **DONE (Rejected: No gain over baseline)**
    - [x] Sigma Sensitivity Analysis (Sigma=1 vs 2). -> **DONE (Sigma=2 Verified as Optimal)**
