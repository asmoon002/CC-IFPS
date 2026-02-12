
import os
import pandas as pd
import glob
import re
import numpy as np

# Paths
PROPOSED_RESULTS = "/userHome/userhome4/sehoon/patchcore/results/project"
BASELINE_RESULTS = "/userHome/userhome4/sehoon/patchcore-inspection-main/results/baseline_comparison"

def parse_results(result_dir, method_name):
    data = []
    if not os.path.exists(result_dir):
        print(f"Warning: Results directory not found: {result_dir}")
        return pd.DataFrame()

    # Walk through directory
    file_count = 0
    for root, dirs, files in os.walk(result_dir):
        if "results.csv" in files:
            csv_path = os.path.join(root, "results.csv")
            folder_name = os.path.basename(root)
            
            # Filter based on method expectations
            if "Proposed" in method_name and not folder_name.startswith("GoldenConfig"):
                continue
            if "Baseline" in method_name and not folder_name.startswith("Baseline"):
                continue

            # Simple parsing of folder name for metadata
            # Expected: GoldenConfig_bottle_Seed0... or Baseline_bottle_Seed0...
            parts = folder_name.split('_')
            if len(parts) < 3:
                continue
                
            class_name = parts[1]
            seed_str = [p for p in parts if p.startswith("Seed")]
            if seed_str:
                seed = int(seed_str[0].replace("Seed", ""))
            else:
                seed = 0
                
            try:
                df = pd.read_csv(csv_path)
                # AP key mapping
                possible_ap_keys = ["Full Pixel AP", "full_pixel_ap", "image_auroc", "instance_auroc"]
                col = next((c for c in possible_ap_keys if c in df.columns), None)
                
                if col:
                    ap = df[col].iloc[0]
                    
                    # Memory key mapping
                    mem_key = "Memory Bank Size" if "Memory Bank Size" in df.columns else "memory_bank_size"
                    mem = df[mem_key].iloc[0] if mem_key in df.columns else 0

                    data.append({
                        "Method": method_name,
                        "Class": class_name,
                        "Seed": seed,
                        "AP": ap,
                        "Memory": mem
                    })
                    file_count += 1
            except Exception as e:
                pass
    
    print(f"[{method_name}] Found {file_count} valid result files in {result_dir}")
    return pd.DataFrame(data)

def main():
    print("Parsing Proposed Results...")
    df_proposed = parse_results(PROPOSED_RESULTS, "Proposed (Hybrid)")
    print("Parsing Baseline Results...")
    df_baseline = parse_results(BASELINE_RESULTS, "Baseline (Greedy 10%)")

    
    if df_proposed.empty and df_baseline.empty:
        print("No results found.")
        return

    df = pd.concat([df_proposed, df_baseline])
    
    # -------------------------------------------------------------------------
    # AP Comparison
    # -------------------------------------------------------------------------
    print("\n=======================================================")
    print("COMPARISON: Proposed vs Baseline (Mean AP)")
    print("=======================================================")
    print(f"{'Class':<12} | {'Proposed':<9} | {'Baseline':<9} | {'Gap':<6} | {'Status'}")
    print("-" * 80)
    
    classes = sorted(list(set(df["Class"])))
    
    total_prop_ap = []
    total_base_ap = []
    
    for cls in classes:
        sub = df[df["Class"] == cls]
        
        prop = sub[sub["Method"] == "Proposed (Hybrid)"]
        base = sub[sub["Method"] == "Baseline (Greedy 10%)"]
        
        mean_prop = prop["AP"].mean() if not prop.empty else 0
        mean_base = base["AP"].mean() if not base.empty else 0
        
        if not prop.empty: total_prop_ap.append(mean_prop)
        if not base.empty: total_base_ap.append(mean_base)
        
        gap = mean_prop - mean_base
        status = "WIN" if gap > 0 else "LOSS" if gap < 0 else "TIE"
        if mean_prop == 0 or mean_base == 0: status = "WAIT"
        
        print(f"{cls:<12} | {mean_prop:.4f}    | {mean_base:.4f}    | {gap:+.4f} | {status}")
        
    print("-" * 80)
    if total_prop_ap and total_base_ap:
        print(f"Total Avg    | {np.mean(total_prop_ap):.4f}    | {np.mean(total_base_ap):.4f}")
    print("=======================================================")

    # -------------------------------------------------------------------------
    # Memory Comparison
    # -------------------------------------------------------------------------
    print("\n=======================================================")
    print("COMPARISON: Proposed vs Baseline (Memory Bank Size)")
    print("=======================================================")
    print(f"{'Class':<12} | {'Proposed':<9} | {'Baseline':<9} | {'Diff':<7} | {'Ratio'}")
    print("-" * 80)

    total_prop_mem = []
    total_base_mem = []

    for cls in classes:
        sub = df[df["Class"] == cls]
        
        prop = sub[sub["Method"] == "Proposed (Hybrid)"]
        base = sub[sub["Method"] == "Baseline (Greedy 10%)"]
        
        mean_prop = prop["Memory"].mean() if not prop.empty else 0
        mean_base = base["Memory"].mean() if not base.empty else 0
        
        if not prop.empty: total_prop_mem.append(mean_prop)
        if not base.empty: total_base_mem.append(mean_base)
        
        diff = mean_prop - mean_base
        ratio = mean_prop / mean_base if mean_base > 0 else 0
        
        print(f"{cls:<12} | {int(mean_prop):<9} | {int(mean_base):<9} | {diff:+7.0f} | {ratio:.1f}x")
        
    print("-" * 80)
    if total_prop_mem and total_base_mem:
        global_prop_mem = np.mean(total_prop_mem)
        global_base_mem = np.mean(total_base_mem)
        print(f"Total Avg    | {int(global_prop_mem):<9} | {int(global_base_mem):<9} | {global_prop_mem - global_base_mem:+7.0f} | {global_prop_mem/global_base_mem:.1f}x")
    print("=======================================================")

if __name__ == "__main__":
    main()
