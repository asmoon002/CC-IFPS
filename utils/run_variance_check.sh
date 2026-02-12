#!/bin/bash
set -e

# Target Classes for Variance Check
# Carpet: Tau=0.01, Budget=50k, L1+2+3, k=1,3,5,9 (Target Mean 0.7042)
# Screw: Tau=0.02, Budget=50k, L1+2+3, d2, k=1 (Target Mean 0.5418)
# Grid: Tau=0.01, Budget=7k, L1+2+3, d2, k=1,3,5,9 (Target Mean 0.4034)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate patchcore

declare -A BEST_CONFIGS
BEST_CONFIGS=(
    ["carpet"]="0.01:50000:layer1,layer2,layer3:1,3,5,9:greedy:none"
    ["screw"]="0.02:50000:layer1,layer2,layer3:1:d2:none"
    ["grid"]="0.01:7000:layer1,layer2,layer3:1,3,5,9:d2:none"
)

# Run 3 seeds
# Explicit run order to minimize interference?
# We will run ONE CLASS at a time for ALL 3 SEEDS to ensure consistent environment per class.
# e.g. Carpet Seed 0, Carpet Seed 1, Carpet Seed 2.

for class_name in "carpet" "screw" "grid"; do
    echo ""
    echo "############################################"
    echo "Starting Variance Check for: $class_name"
    echo "############################################"
    
    config="${BEST_CONFIGS[$class_name]}"
    IFS=':' read -r tau budget layers knn sampling_type postprocess <<< "$config"
    
    if [ -z "$postprocess" ]; then postprocess="none"; fi
    
    layer_args=""
    IFS=',' read -ra LAYER_ARRAY <<< "$layers"
    for layer in "${LAYER_ARRAY[@]}"; do
        layer_args="$layer_args -le $layer"
    done

    for seed in 0 1 2; do
        echo ""
        echo "=== Run Seed $seed for $class_name ==="
        
        # Explicit sleep to clean GPU
        sleep 5
        
        python bin/run_patchcore.py \
            --gpu 1 --seed $seed \
            --log_group "Variance_Check_${class_name}_Run${seed}" \
            results \
            patch_core \
            -b wideresnet50 \
            $layer_args \
            --faiss_on_gpu \
            --pretrain_embed_dimension 1024 \
            --target_embed_dimension 1024 \
            --anomaly_scorer_num_nn 1 \
            --multiscale_knn $knn \
            --patchsize 3 \
            --use_density_weighted_scoring \
            --postprocess $postprocess \
            sampler \
            --sampling_type $sampling_type \
            --tau $tau \
            --max_memory_size $budget \
            --use_hybrid \
            cc_ifps \
            dataset \
            --resize 256 --imagesize 224 \
            --augment \
            -d $class_name \
            mvtec /userHome/userhome4/sehoon/patchcore/data
            
        echo "=== Completed Seed $seed for $class_name ==="
    done
done

echo ""
echo "All Variance Runs Complete. Parsing results..."
# Simple parser to output the values directly
echo "------------------------------------------------"
echo "Results Summary:"
grep "Anomaly Pixel AP" results/project/Variance_Check_*/results.csv | sort
