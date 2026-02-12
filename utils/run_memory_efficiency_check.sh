#!/bin/bash

# Validating environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate patchcore
export PYTHONNOUSERSITE=1
export PYTHONPATH=src

# =========================================================================================
# MEMORY EFFICIENCY CHECK: ISO-MEMORY BUDGET (20k)
# =========================================================================================
# Goal: Prove that Hybrid PatchCore achieves higher AP than Baseline given the SAME memory budget.
# Baseline averages ~19k patches. We set Hybrid budget to 20k to make a fair comparison.
# =========================================================================================

# Set GPU ID
GPU_ID=0

# Dataset Path
DATASET_PATH="/userHome/userhome4/sehoon/patchcore/data"

# Results Path
RESULTS_PATH="results_memory_efficiency_check_20k"

# Log Name Suffix
LOG_NAME_PREFIX="MemoryCheck"

# Parameters
KNN=1
FIXED_BUDGET=20000  # Matching Baseline's ~19k average

# Define seed 1 only for quick validation
SEEDS=(1)

echo "Starting Memory Efficiency Check (Budget=${FIXED_BUDGET}) on GPU ${GPU_ID}..."
echo "Results will be saved to ${RESULTS_PATH}"

# Loop through all MVTec classes
for CLASS in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do
    
    # Use the same best configurations as Golden, but override BUDGET
    case "$CLASS" in
        "bottle")       SAMPLER="greedy" ; LAYERS="-le layer2 -le layer3"       ; LAYER_TAG="L2-3"   ; TAU=0.01 ;;
        "cable")        SAMPLER="greedy" ; LAYERS="-le layer2 -le layer3"       ; LAYER_TAG="L2-3"   ; TAU=0.02 ;;
        "capsule")      SAMPLER="greedy" ; LAYERS="-le layer2 -le layer3"       ; LAYER_TAG="L2-3"   ; TAU=0.03 ;;
        "carpet")       SAMPLER="d2"     ; LAYERS="-le layer1 -le layer2 -le layer3" ; LAYER_TAG="L1-2-3" ; TAU=0.03 ;;
        "grid")         SAMPLER="d2"     ; LAYERS="-le layer1 -le layer2 -le layer3" ; LAYER_TAG="L1-2-3" ; TAU=0.02 ;;
        "hazelnut")     SAMPLER="greedy" ; LAYERS="-le layer1 -le layer2 -le layer3" ; LAYER_TAG="L1-2-3" ; TAU=0.02 ;;
        "leather")      SAMPLER="greedy" ; LAYERS="-le layer1 -le layer2 -le layer3" ; LAYER_TAG="L1-2-3" ; TAU=0.02 ;;
        "metal_nut")    SAMPLER="greedy" ; LAYERS="-le layer1 -le layer2 -le layer3" ; LAYER_TAG="L1-2-3" ; TAU=0.03 ;;
        "pill")         SAMPLER="greedy" ; LAYERS="-le layer1 -le layer2 -le layer3" ; LAYER_TAG="L1-2-3" ; TAU=0.01 ;;
        "screw")        SAMPLER="d2"     ; LAYERS="-le layer1 -le layer2 -le layer3" ; LAYER_TAG="L1-2-3" ; TAU=0.01 ;;
        "tile")         SAMPLER="greedy" ; LAYERS="-le layer1 -le layer2 -le layer3" ; LAYER_TAG="L1-2-3" ; TAU=0.02 ;;
        "toothbrush")   SAMPLER="d2"     ; LAYERS="-le layer2 -le layer3"       ; LAYER_TAG="L2-3"   ; TAU=0.01 ;;
        "transistor")   SAMPLER="greedy" ; LAYERS="-le layer2 -le layer3"       ; LAYER_TAG="L2-3"   ; TAU=0.02 ;;
        "wood")         SAMPLER="greedy" ; LAYERS="-le layer1 -le layer2 -le layer3" ; LAYER_TAG="L1-2-3" ; TAU=0.01 ;;
        "zipper")       SAMPLER="greedy" ; LAYERS="-le layer1 -le layer2 -le layer3" ; LAYER_TAG="L1-2-3" ; TAU=0.01 ;;
    esac

    BUDGET_K=$((FIXED_BUDGET / 1000))

    # Run Seed 1
    for SEED in "${SEEDS[@]}"; do
        
        LOG_NAME="${LOG_NAME_PREFIX}_${CLASS}_Seed${SEED}_${SAMPLER}_tau${TAU}_${LAYER_TAG}_mem${BUDGET_K}k"
        
        echo "Running ${CLASS}: Seed=${SEED}, ${SAMPLER^^}, τ=${TAU}, ${LAYER_TAG}, B=${BUDGET_K}k"
        
        /userHome/userhome4/sehoon/miniconda3/envs/patchcore/bin/python bin/run_patchcore.py \
            --gpu $GPU_ID --seed $SEED \
            --log_group "$LOG_NAME" \
            results_memory_efficiency_check_20k \
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
            --max_memory_size $FIXED_BUDGET \
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
echo "✅ Memory Efficiency Benchmark Complete!"
echo "Comparing results..."
python compare_memory_efficiency.py
echo "============================================"
