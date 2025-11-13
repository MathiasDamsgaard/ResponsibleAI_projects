#!/bin/bash
#BSUB -J protopnet_200proto
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

# =============================================================================
# EDIT THESE PARAMETERS BEFORE SUBMITTING
# =============================================================================

# Required parameters
DATASET="./datasets/CUB_200_2011"  # Path to dataset directory (contains train/ and test/)
EXP_NAME="protopnet_200proto"     # Experiment name

# Training parameters
EPOCHS=200                          # Number of training epochs
WARM_EPOCHS=50                      # Number of warm-up epochs
TEST_INTERVAL=50                    # Test every N epochs
PUSH_INTERVAL=50                    # Push prototypes every N epochs
NUM_PROTOTYPES=200                  # Number of prototypes to learn

# Model architecture parameters
ARCHITECTURE="resnet34"             # Backbone architecture
BATCH_SIZE=32                       # Batch size
IMG_SIZE=224                        # Image size
STEP_SIZE=150                       # Learning rate scheduler step size

# Diversity regularization parameters
MIN_DIVERSITY=0.1                   # Minimum diversity
DIVERSITY_COEFF=0.1                 # Diversity coefficient

# Other parameters
PROTOTYPE_ACTIVATION="log"          # Prototype activation function
ADD_ON_LAYERS="regular"             # Add-on layers type
NUM_WORKERS=2                       # Number of data loading workers

# =============================================================================
# DO NOT EDIT BELOW THIS LINE
# =============================================================================

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate Python environment
if [ -d "../../env" ]; then
  source ../../env/bin/activate
elif [ -d "../env" ]; then
  source ../env/bin/activate
elif [ -d "env" ]; then
  source env/bin/activate
fi

echo "=========================================="
echo "ProtoPNet Training Job"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Experiment name: $EXP_NAME"
echo "Epochs: $EPOCHS"
echo "Warm epochs: $WARM_EPOCHS"
echo "Test interval: $TEST_INTERVAL"
echo "Push interval: $PUSH_INTERVAL"
echo "Number of prototypes: $NUM_PROTOTYPES"
echo "Architecture: $ARCHITECTURE"
echo "Batch size: $BATCH_SIZE"
echo "Image size: $IMG_SIZE"
echo "Step size: $STEP_SIZE"
echo "Min diversity: $MIN_DIVERSITY"
echo "Diversity coefficient: $DIVERSITY_COEFF"
echo "Prototype activation: $PROTOTYPE_ACTIVATION"
echo "Add-on layers: $ADD_ON_LAYERS"
echo "Number of workers: $NUM_WORKERS"
echo "Seed: ${SEED:-Not set}"
echo "=========================================="

# Run the training
echo "Running command:"
echo "python train.py --dataset $DATASET --exp_name $EXP_NAME --epochs $EPOCHS ..."
echo "=========================================="

python train.py \
  --dataset "$DATASET" \
  --exp_name "$EXP_NAME" \
  --epochs $EPOCHS \
  --warm_epochs $WARM_EPOCHS \
  --test_interval $TEST_INTERVAL \
  --push_interval $PUSH_INTERVAL \
  --num_prototypes $NUM_PROTOTYPES \
  --architecture $ARCHITECTURE \
  --batch_size $BATCH_SIZE \
  --img_size $IMG_SIZE \
  --step_size $STEP_SIZE \
  --min_diversity $MIN_DIVERSITY \
  --diversity_coeff $DIVERSITY_COEFF \
  --prototype_activation_function $PROTOTYPE_ACTIVATION \
  --add_on_layers $ADD_ON_LAYERS \
  --num_workers $NUM_WORKERS

echo "=========================================="
echo "Training completed!"
echo "=========================================="
