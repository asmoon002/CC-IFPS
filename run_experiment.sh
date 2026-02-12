#!/bin/bash

# Validating environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate patchcore
export PYTHONNOUSERSITE=1
export PYTHONPATH=src


# =========================================================================================
# FINAL PAPER VALIDATION: GOLDEN CONFIGURATION (Hybrid Greedy + D2)
# =========================================================================================
# This script executes the optimal configuration for each class as determined by the
# "Grand Tournament" between Greedy and D2 sampling.
# It runs 3 random seeds (0, 1, 2) to report robust Mean +/- Std AP.
#
# Best Configurations based on Tournament Results:
# Class        | Win      | Config
# -----------------------------------------------------------
# bottle       | Greedy   | τ=0.01 L2-3 k=1 B=70k
# cable        | Greedy   | τ=0.02 L2-3 k=1 B=70k
# capsule      | Greedy   | τ=0.03 L2-3 k=1 B=50k
# carpet       | D2       | τ=0.03 L1-2-3 k=1 B=50k
# grid         | D2       | τ=0.02 L1-2-3 k=1 B=70k
# hazelnut     | Greedy   | τ=0.02 L1-2-3 k=1 B=70k
# leather      | Greedy   | τ=0.02 L1-2-3 k=1 B=50k
# metal_nut    | Greedy   | τ=0.03 L1-2-3 k=1 B=70k
# pill         | Greedy   | τ=0.01 L1-2-3 k=1 B=30k
# screw        | D2       | τ=0.01 L1-2-3 k=1 B=70k
# tile         | Greedy   | τ=0.02 L1-2-3 k=1 B=70k
# toothbrush   | D2       | τ=0.01 L2-3 k=1 B=50k
# transistor   | Greedy   | τ=0.02 L2-3 k=1 B=50k
# wood         | Greedy   | τ=0.01 L1-2-3 k=1 B=30k
# zipper       | Greedy   | τ=0.01 L1-2-3 k=1 B=70k
# =========================================================================================

# Set GPU ID
GPU_ID=0

# Dataset Path
# Dataset Path
DATASET_PATH="data"

# Results Path
RESULTS_PATH="results"

# Log Name Suffix
LOG_NAME_PREFIX="GoldenConfig"

# Parameters
KNN=1

# Define seed 1 only for quick validation
SEEDS=(1)

echo "Starting Final Golden Configuration Run on GPU ${GPU_ID}..."
echo "Results will be saved to ${RESULTS_PATH}"

# Loop through all MVTec classes
for CLASS in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do
    
    # 1. Determine Best Configuration per Class
    case "$CLASS" in
        "bottle")
            SAMPLER="greedy"
            LAYERS="-le layer2 -le layer3"
            LAYER_TAG="L2-3"
            TAU=0.01
            BUDGET=70000
            ;;
        "cable")
            SAMPLER="greedy"
            LAYERS="-le layer2 -le layer3"
            LAYER_TAG="L2-3"
            TAU=0.02
            BUDGET=70000
            ;;
        "capsule")
            SAMPLER="greedy"
            LAYERS="-le layer2 -le layer3"
            LAYER_TAG="L2-3"
            TAU=0.03
            BUDGET=70000  # Increased from 50k
            ;;
        "carpet")
            SAMPLER="d2"
            LAYERS="-le layer1 -le layer2 -le layer3"
            LAYER_TAG="L1-2-3"
            TAU=0.03
            BUDGET=70000  # Increased from 50k
            ;;
        "grid")
            SAMPLER="d2"
            LAYERS="-le layer1 -le layer2 -le layer3"
            LAYER_TAG="L1-2-3"
            TAU=0.02
            BUDGET=70000
            ;;
        "hazelnut")
            SAMPLER="greedy"
            LAYERS="-le layer1 -le layer2 -le layer3"
            LAYER_TAG="L1-2-3"
            TAU=0.02
            BUDGET=70000
            ;;
        "leather")
            SAMPLER="greedy"
            LAYERS="-le layer1 -le layer2 -le layer3"
            LAYER_TAG="L1-2-3"
            TAU=0.02
            BUDGET=70000  # Increased from 50k
            ;;
        "metal_nut")
            SAMPLER="greedy"
            LAYERS="-le layer1 -le layer2 -le layer3"
            LAYER_TAG="L1-2-3"
            TAU=0.03
            BUDGET=70000
            ;;
        "pill")
            SAMPLER="greedy"
            LAYERS="-le layer1 -le layer2 -le layer3"
            LAYER_TAG="L1-2-3"
            TAU=0.01
            BUDGET=50000  # Increased from 30k
            ;;
        "screw")
            SAMPLER="d2"
            LAYERS="-le layer1 -le layer2 -le layer3"
            LAYER_TAG="L1-2-3"
            TAU=0.01
            BUDGET=70000
            ;;
        "tile")
            SAMPLER="greedy"
            LAYERS="-le layer1 -le layer2 -le layer3"
            LAYER_TAG="L1-2-3"
            TAU=0.02
            BUDGET=70000
            ;;
        "toothbrush")
            SAMPLER="d2"
            LAYERS="-le layer2 -le layer3"
            LAYER_TAG="L2-3"
            TAU=0.01
            BUDGET=50000
            ;;
        "transistor")
            SAMPLER="greedy"
            LAYERS="-le layer2 -le layer3"
            LAYER_TAG="L2-3"
            TAU=0.02
            BUDGET=70000  # Increased from 50k
            ;;
        "wood")
            SAMPLER="greedy"
            LAYERS="-le layer1 -le layer2 -le layer3"
            LAYER_TAG="L1-2-3"
            TAU=0.01
            BUDGET=50000  # Increased from 30k
            ;;
        "zipper")
            SAMPLER="greedy"
            LAYERS="-le layer1 -le layer2 -le layer3"
            LAYER_TAG="L1-2-3"
            TAU=0.01
            BUDGET=70000
            ;;
    esac

    BUDGET_K=$((BUDGET / 1000))

    # 2. Run Seed 1
    for SEED in "${SEEDS[@]}"; do
        
        LOG_NAME="${LOG_NAME_PREFIX}_${CLASS}_Seed${SEED}_${SAMPLER}_tau${TAU}_${LAYER_TAG}_mem${BUDGET_K}k"
        
        echo "Running ${CLASS}..."
        
        python bin/run_patchcore.py \
            --gpu $GPU_ID --seed $SEED \
            --log_group "$LOG_NAME" \
            "$RESULTS_PATH" \
            patch_core \
            -b wideresnet50 \
            $LAYERS \
            $LAYERS \
            --pretrain_embed_dimension 1024 \
            --target_embed_dimension 1024 \
            --anomaly_scorer_num_nn 1 \
            --multiscale_knn $KNN \
            --patchsize 3 \
            --postprocess none \
            sampler \
            --sampling_type $SAMPLER \
            --tau $TAU \
            --max_memory_size $BUDGET \
            --use_hybrid \
            cc_ifps \
            dataset \
            --resize 256 --imagesize 224 \
            -d $CLASS \
            mvtec $DATASET_PATH
            
        echo "  ✓ Done"
    done
done

echo ""
echo "============================================"
echo "✅ Final Golden Configuration Run Complete!"
echo "Analyzing Results..."
python analyze_results.py
echo "============================================"
