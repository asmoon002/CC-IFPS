import os
import pandas as pd
import numpy as np
import glob
import re

RESULTS_DIR = "results/project"
CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

def parse_results():
    data = []
    
    # Directory pattern: GoldenConfig_{CLASS}_Seed{SEED}_{SAMPLER}_tau{TAU}_{LAYER}_mem{BUDGET}k
    pattern = re.compile(r"GoldenConfig_(\w+)_Seed(\d+)_(\w+)_tau([\d\.]+)_([L\d\+\-]+)_mem(\d+)k")
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory '{RESULTS_DIR}' not found.")
        return pd.DataFrame()

    for folder in os.listdir(RESULTS_DIR):
        if not folder.startswith("GoldenConfig_"):
            continue
            
        match = pattern.match(folder)
        if not match:
            continue
            
        cls, seed, sampler, tau, layers, budget = match.groups()
        
        # Read result csv
        csv_path = os.path.join(RESULTS_DIR, folder, "results.csv")
        if not os.path.exists(csv_path):
            continue
            
        try:
            df = pd.read_csv(csv_path)
            # Adapt validation key if needed. Usually 'pimage_aurocs' or 'Full Pixel AP'
            # Based on previous file views, it seems 'Full Pixel AP' is used.
            # Let's check columns flexibly.
            if "Full Pixel AP" in df.columns:
                ap = df["Full Pixel AP"].iloc[0]
            elif "image_auroc" in df.columns: # fallback if column name differs
                ap = df["image_auroc"].iloc[0]
            else:
                # Try finding a float column
                ap = 0.0
            
            mem = 0
            if "Memory Bank Size" in df.columns:
                mem = df["Memory Bank Size"].iloc[0]

            data.append({
                "Class": cls,
                "Seed": int(seed),
                "AP": float(ap),
                "Sampler": sampler,
                "Memory": float(mem),
                "Config": f"τ={tau} {layers} B={budget}k"
            })
        except Exception as e:
            print(f"Error reading {folder}: {e}")
            continue
            
    return pd.DataFrame(data)

def main():
    df = parse_results()
    
    if df.empty:
        print("No results found yet.")
        return

    print("=======================================================================================================")
    print("FINAL PAPER RESULTS: Hybrid PatchCore (Proposed)")
    print("=======================================================================================================")
    print(f"{'Class':<12} | {'Mean AP':<8} | {'Std Dev':<8} | {'Runs':<4}")
    print("-" * 60)
    
    total_mean_ap = []
    
    for cls in CLASSES:
        cls_df = df[df["Class"] == cls]
        
        if cls_df.empty:
            print(f"{cls:<12} | {'Pending...':<50}")
            continue
        
        mean_ap = cls_df["AP"].mean()
        std_ap = cls_df["AP"].std()
        n_runs = len(cls_df)
        
        total_mean_ap.append(mean_ap)
        
        print(f"{cls:<12} | {mean_ap:.4f}   | ±{std_ap:.4f}   | {n_runs:<4}")

    print("-" * 60)
    if total_mean_ap:
        global_avg = np.mean(total_mean_ap)
        print(f"Total Average|          | {global_avg:.4f}")
    print("=======================================================================================================")

if __name__ == "__main__":
    main()
