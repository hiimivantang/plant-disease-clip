# Plant Disease CLIP Fine-tuning

A fine-tuning implementation for OpenAI's CLIP model specialized for plant disease classification. This project uses supervised contrastive learning to adapt CLIP for accurate plant disease identification while maintaining generalization capabilities.

## Overview

This project fine-tunes CLIP (Contrastive Language-Image Pre-training) models to classify plant diseases from images. It includes two training modes:
- **Original Mode**: Aggressive fine-tuning optimized for the 38 training classes
- **Generalized Mode**: Conservative fine-tuning with better generalization to unseen plant species

**Pre-trained Model Available**: A fine-tuned CLIP ViT-L-14 model (768-dimensional embeddings) is available on HuggingFace: [hiimivantang/plant-disease-clip-768dim](https://huggingface.co/hiimivantang/plant-disease-clip-768dim)

## Features

- Supervised contrastive learning with CLIP
- Dual training modes (original vs generalized)
- Support for multiple plant species and disease types
- Easy dataset expansion with new plant-disease pairs
- Checkpoint saving and model evaluation
- Integration-ready for vector database applications
- Pre-trained ViT-L-14 model available on HuggingFace

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Conda or virtualenv

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd plant-disease-clip
```

2. Create and activate a conda environment:
```bash
conda create -n finetuning python=3.9
conda activate finetuning
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Using Pre-trained Model

A fine-tuned CLIP ViT-L-14 model is available on HuggingFace and ready to use without training:

**Model**: [hiimivantang/plant-disease-clip-768dim](https://huggingface.co/hiimivantang/plant-disease-clip-768dim)

### Quick Start with Pre-trained Model

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load the pre-trained model from HuggingFace
model = CLIPModel.from_pretrained("hiimivantang/plant-disease-clip-768dim")
processor = CLIPProcessor.from_pretrained("hiimivantang/plant-disease-clip-768dim")

# Load and process an image
image = Image.open("path/to/plant_image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

# image_features is a 768-dimensional embedding ready for similarity search
```

**Model Details:**
- Architecture: ViT-L-14
- Embedding dimension: 768
- Fine-tuned on plant disease dataset
- Ready for inference and vector database integration

If you want to fine-tune your own model or modify the training process, continue with the sections below.

## Dataset Structure

The dataset follows a structured format in the `kaggle-data/` directory:

```
kaggle-data/
├── train/
│   ├── PlantName___DiseaseName/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── PlantName___healthy/
│   │   └── ...
│   └── ...
├── valid/
│   └── (same structure as train/)
└── test/
    └── (same structure as train/)
```

### Naming Convention

Each subdirectory must follow this exact pattern:
- Format: `PlantName___DiseaseName`
- Use **three underscores** (`___`) as separator
- For healthy plants: `PlantName___healthy`
- Examples:
  - `Apple___Apple_scab`
  - `Tomato___Early_blight`
  - `Corn_(maize)___Common_rust_`
  - `Grape___healthy`

## Adding New Plant-Disease Pairs

To expand the dataset with new plant-disease combinations:

### Step 1: Prepare Your Images

Collect images for the new plant-disease pair. Recommended:
- Minimum 50-100 images per class for training
- 10-20 images for validation
- 10-20 images for testing
- Images should be clear and focused on disease symptoms
- Supported formats: JPG, JPEG, PNG

### Step 2: Create Directory Structure

1. Navigate to the appropriate directory:
```bash
cd kaggle-data/train
```

2. Create a new directory following the naming convention:
```bash
mkdir "PlantName___DiseaseName"
# Example:
mkdir "Tomato___Late_blight"
```

3. Repeat for validation and test sets:
```bash
cd ../valid
mkdir "Tomato___Late_blight"

cd ../test
mkdir "Tomato___Late_blight"
```

### Step 3: Add Images

Copy your images into the newly created directories:
```bash
# Copy training images
cp /path/to/your/images/*.jpg kaggle-data/train/Tomato___Late_blight/

# Copy validation images
cp /path/to/your/validation/images/*.jpg kaggle-data/valid/Tomato___Late_blight/

# Copy test images
cp /path/to/your/test/images/*.jpg kaggle-data/test/Tomato___Late_blight/
```

### Step 4: Verify Structure

Check that your directory structure is correct:
```bash
# Should list your new directory
ls kaggle-data/train/ | grep "Tomato___Late_blight"

# Verify images are present
ls kaggle-data/train/Tomato___Late_blight/ | wc -l
```

### Step 5: Update and Train

The training script automatically detects all subdirectories, so no code changes are needed. Simply run the training process (see below).

### Important Notes

- **Directory naming**: The three underscores (`___`) separator is critical
- **Image quality**: Higher quality images lead to better model performance
- **Class balance**: Try to maintain similar numbers of images across classes
- **Data splits**: Maintain roughly 70-20-10 split (train-valid-test)

## Starting Fine-tuning

### Quick Start

Run training with the default generalized mode:
```bash
bash run_training.sh
```

### Training Modes

#### 1. Generalized Mode (Recommended)
Best for diverse datasets and better generalization to unseen plants:
```bash
bash run_training.sh generalized
```

**Characteristics:**
- Epochs: 5
- Learning rate: 5e-5
- Includes regularization
- Better zero-shot performance on new plants
- More conservative fine-tuning

#### 2. Original Mode
Aggressive fine-tuning optimized for the training classes:
```bash
bash run_training.sh original
```

**Characteristics:**
- Epochs: 10
- Learning rate: 1e-4
- Higher accuracy on training classes
- May overfit on limited data

### Manual Training

You can also run training scripts directly:

```bash
# Generalized mode
python finetune/train_supcon_clip_generalized.py

# Original mode
python finetune/train_supcon_clip_fixed.py
```

### Training Output

During training, you'll see:
- Loss metrics per epoch
- Validation accuracy
- Checkpoint saving notifications

Checkpoints are saved to `checkpoints/`:
- `supcon_clip_*_best.pt` - Best model based on validation accuracy
- `supcon_clip_*_final.pt` - Final model after all epochs

## Project Structure

```
plant-disease-clip/
├── kaggle-data/           # Training dataset
│   ├── train/            # Training images
│   ├── valid/            # Validation images
│   └── test/             # Test images
├── finetune/             # Training scripts
│   ├── train_supcon_clip_fixed.py          # Original training mode
│   ├── train_supcon_clip_generalized.py    # Generalized training mode
│   └── GENERALIZATION_GUIDE.md             # Detailed training guide
├── checkpoints/          # Saved model checkpoints (created during training)
├── run_training.sh       # Main training launcher script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Using the Trained Model

After training, you can use the model for:

1. **Inference**: Load the checkpoint and classify new plant images
2. **Embedding extraction**: Generate embeddings for vector database applications
3. **Transfer learning**: Use as a base model for related tasks

Example loading a checkpoint:
```python
import torch
from open_clip import create_model_and_transforms

# Load model
model, preprocess_train, preprocess_val = create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)

# Load fine-tuned weights
checkpoint = torch.load('checkpoints/supcon_clip_generalized_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference or embedding extraction
```

## Troubleshooting

### Data Not Found Error
```
⚠  Training data not found at kaggle-data/train
```
**Solution**: Ensure the `kaggle-data/train` and `kaggle-data/valid` directories exist and contain subdirectories with images.

### Out of Memory
**Solution**: Reduce batch size in the training script or use a GPU with more memory.

### Poor Generalization
**Solution**: Use the generalized training mode and ensure diverse training data.

### Conda Environment Issues
**Solution**: Edit `run_training.sh` line 23 to match your conda installation path and environment name.

## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Add your plant-disease pairs or improvements
4. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- OpenAI CLIP model
- OpenCLIP implementation
- Plant disease dataset contributors
- Kaggle plant disease datasets

## Citation

If you use this project in your research, please cite:
```
[Add citation information here]
```

## Contact

[Add contact information here]

## References

- [OpenAI CLIP Paper](https://arxiv.org/abs/2103.00020)
- [OpenCLIP Repository](https://github.com/mlfoundations/open_clip)
- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
