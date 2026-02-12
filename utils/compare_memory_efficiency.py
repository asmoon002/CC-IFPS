import pandas as pd
import os
import numpy as np

# Define Paths
BASELINE_RESULTS = "/userHome/userhome4/sehoon/patchcore-inspection-main/results/baseline_comparison"
ISO_MEMORY_RESULTS = "/userHome/userhome4/sehoon/patchcore/results_memory_efficiency_check_20k"

def parse_results(result_dir, method_name):
    data = []
    
    if not os.path.exists(result_dir):
        print(f"Directory {result_dir} not found.")
        return pd.DataFrame()

    file_count = 0
    for root, dirs, files in os.walk(result_dir):
        if "results.csv" in files:
            try:
                # Extract Class Name from folder name
                folder_name = os.path.basename(root)
                parts = folder_name.split("_")
                
                # Heuristic parsing of class name
                if "Baseline" in folder_name:
                    # Format: Baseline_{CLASS}_Seed...
                    class_name = parts[1]
                    if class_name == "metal": class_name = "metal_nut" # Fix metal_nut
                elif "MemoryCheck" in folder_name:
                    # Format: MemoryCheck_{CLASS}_Seed...
                    class_name = parts[1]
                    if class_name == "metal": class_name = "metal_nut"
                else:
                    continue

                csv_path = os.path.join(root, "results.csv")
                df = pd.read_csv(csv_path)
                
                # STRICT MODE: Only accept 'full_pixel_ap' or 'Full Pixel AP'
                possible_ap_keys = ["Full Pixel AP", "full_pixel_ap"]
                col = next((c for c in possible_ap_keys if c in df.columns), None)
                
                # Memory Extraction
                mem_key = "Memory Bank Size" if "Memory Bank Size" in df.columns else "memory_bank_size"

                if col:
                    ap = df[col].iloc[0]
                    mem = df[mem_key].iloc[0] if mem_key in df.columns else 0
                    
                    data.append({
                        "Method": method_name,
                        "Class": class_name,
                        "AP": ap,
                        "Memory": mem
                    })
                    file_count += 1
            except Exception as e:
                pass
    
    print(f"[{method_name}] Found {file_count} valid result files.")
    return pd.DataFrame(data)

def main():
    print("Parsing Iso-Memory (20k) Hybrid Results...")
    df_hybrid = parse_results(ISO_MEMORY_RESULTS, "Hybrid (20k)")
    
    print("Parsing Baseline Results...")
    df_baseline = parse_results(BASELINE_RESULTS, "Baseline (Standard)")
    
    if df_hybrid.empty and df_baseline.empty:
        print("No results found.")
        return

    df = pd.concat([df_hybrid, df_baseline])
    
    print("\n=======================================================================")
    print("PERFORMANCE BENCHMARK: Hybrid (20k) vs Baseline (Standard)")
    print("=======================================================================")
    print(f"{'Class':<12} | {'AP (Hybrid)':<12} | {'AP (Base)':<10} | {'Gap':<6}")
    print("-" * 64)
    
    classes = sorted(list(set(df["Class"])))
    
    total_hybrid_ap = []
    total_base_ap = []
    
    for cls in classes:
        sub = df[df["Class"] == cls]
        
        hybrid = sub[sub["Method"] == "Hybrid (20k)"]
        base = sub[sub["Method"] == "Baseline (Standard)"]
        
        mean_hybrid = hybrid["AP"].mean() if not hybrid.empty else 0
        mean_base = base["AP"].mean() if not base.empty else 0
        
        if not hybrid.empty: total_hybrid_ap.append(mean_hybrid)
        if not base.empty: total_base_ap.append(mean_base)
        
        gap = mean_hybrid - mean_base
        
        print(f"{cls:<12} | {mean_hybrid:.4f}       | {mean_base:.4f}     | {gap:+.4f}")
        
    print("-" * 64)
    if total_hybrid_ap and total_base_ap:
        global_hybrid = np.mean(total_hybrid_ap)
        global_base = np.mean(total_base_ap)
        print(f"Total Avg    | {global_hybrid:.4f}       | {global_base:.4f}     | {global_hybrid - global_base:+.4f}")
    print("=======================================================================")

if __name__ == "__main__":
    main()
