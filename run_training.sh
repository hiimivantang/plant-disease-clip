#!/bin/bash

# Script to run CLIP fine-tuning for plant disease classification
# Usage: bash run_training.sh [original|generalized]

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script choice
SCRIPT_TYPE=${1:-generalized}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CLIP Fine-tuning for Plant Diseases  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate conda environment
echo -e "${GREEN}✓${NC} Activating finetuning environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# Check if data exists
if [ ! -d "kaggle-data/train" ]; then
    echo -e "${YELLOW}⚠${NC}  Training data not found at kaggle-data/train"
    exit 1
fi

if [ ! -d "kaggle-data/valid" ]; then
    echo -e "${YELLOW}⚠${NC}  Validation data not found at kaggle-data/valid"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found training data"
echo -e "${GREEN}✓${NC} Found validation data"
echo ""

# Select script
if [ "$SCRIPT_TYPE" = "original" ]; then
    SCRIPT="finetune/train_supcon_clip_fixed.py"
    echo -e "${BLUE}Training Mode:${NC} Original (Aggressive fine-tuning)"
    echo -e "${BLUE}Best for:${NC} Only the 38 training classes"
    echo -e "${BLUE}Epochs:${NC} 10 | ${BLUE}LR:${NC} 1e-4"
elif [ "$SCRIPT_TYPE" = "generalized" ]; then
    SCRIPT="finetune/train_supcon_clip_generalized.py"
    echo -e "${BLUE}Training Mode:${NC} Generalized (Better for unseen plants)"
    echo -e "${BLUE}Best for:${NC} Diverse dataset with unseen plants"
    echo -e "${BLUE}Epochs:${NC} 5 | ${BLUE}LR:${NC} 5e-5 | ${BLUE}Regularization:${NC} Yes"
else
    echo -e "${YELLOW}⚠${NC}  Invalid option: $SCRIPT_TYPE"
    echo "Usage: bash run_training.sh [original|generalized]"
    exit 1
fi

echo ""
echo -e "${YELLOW}Starting training in 3 seconds...${NC}"
echo -e "${YELLOW}Press Ctrl+C to cancel${NC}"
sleep 3

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Training Started!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Run training
python $SCRIPT

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Training Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Checkpoints saved to:${NC} checkpoints/"
echo -e "${BLUE}Best model:${NC} checkpoints/supcon_clip*_best.pt"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Test the model: python finetune/test_generalization.py"
echo "  2. Use embeddings for your vector database"
echo ""
