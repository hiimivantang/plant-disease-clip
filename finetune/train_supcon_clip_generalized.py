import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import open_clip

# ====================================================
# ✅ 1. Configuration for Better Generalization
# ====================================================
train_dir = "kaggle-data/train"
val_dir = "kaggle-data/valid"
batch_size = 16  # Reduced for ViT-L-14 (large model needs less memory)
accum_steps = 16  # Increased to maintain effective batch size of 256
epochs = 5  # Fewer epochs to avoid overfitting
lr = 5e-5  # Lower learning rate for gentle fine-tuning
temperature = 0.07
embed_dim = 512  # Keep CLIP's original dimension (no projection head)

# Device setup
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

num_workers = 0 if device == "mps" else 4

# ====================================================
# ✅ 2. Validate data directories
# ====================================================
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation directory not found: {val_dir}")

# ====================================================
# ✅ 3. Data pipeline with stronger augmentation
# ====================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # More aggressive crops
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),  # Plant leaves can be upside down
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(30),
    transforms.RandomGrayscale(p=0.1),  # Helps learn shape features
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),  # CLIP's normalization
                        std=(0.26862954, 0.26130258, 0.27577711))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711))
])

try:
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Number of classes: {len(train_ds.classes)}")
except Exception as e:
    raise RuntimeError(f"Failed to load datasets: {e}")

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=(device == "cuda"))
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device == "cuda"))

# ====================================================
# ✅ 4. Model setup - Light fine-tuning approach
# ====================================================
print("Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-L-14", #ViT-B-32
    pretrained="openai"
)

# Extract vision encoder
vision_encoder = model.visual.to(device)

# Store original model for regularization
original_model = open_clip.create_model_and_transforms(
    model_name="ViT-L-14", #ViT-B-32
    pretrained="openai"
)[0].visual.to(device)
original_model.eval()
for param in original_model.parameters():
    param.requires_grad = False

print(f"Model loaded. Vision encoder output dim: {vision_encoder.output_dim}")
print(f"Using full CLIP embedding (no projection head for better generalization)")

# ====================================================
# ✅ 5. Supervised Contrastive Loss with Regularization
# ====================================================
def supervised_contrastive_loss_with_reg(features, labels, original_features,
                                         temperature=0.07, reg_lambda=0.1):
    # Normalize features
    features = F.normalize(features, dim=1)
    original_features = F.normalize(original_features, dim=1)

    # Contrastive loss
    similarity_matrix = torch.matmul(features, features.T)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # Remove self-comparisons
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0)).to(device)
    mask = mask * logits_mask

    # Compute logits
    logits = similarity_matrix / temperature

    # Log probabilities
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

    # Mean log-likelihood over positives
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    contrastive_loss = -mean_log_prob_pos.mean()

    # Regularization: keep embeddings close to original CLIP
    reg_loss = F.mse_loss(features, original_features)

    total_loss = contrastive_loss + reg_lambda * reg_loss

    return total_loss, contrastive_loss, reg_loss

# ====================================================
# ✅ 6. Optimizer with Layer-wise Learning Rates
# ====================================================
# Fine-tune later layers more, keep early layers more stable
params = []
for name, param in vision_encoder.named_parameters():
    if 'ln_post' in name or 'proj' in name:  # Last layers
        params.append({'params': param, 'lr': lr})
    else:  # Earlier layers
        params.append({'params': param, 'lr': lr * 0.1})

optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# ====================================================
# ✅ 7. Validation function
# ====================================================
def validate(vision_encoder, original_model, val_loader, device):
    vision_encoder.eval()
    total_loss = 0.0
    total_cont_loss = 0.0
    total_reg_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            feats = vision_encoder(images)
            original_feats = original_model(images)
            loss, cont_loss, reg_loss = supervised_contrastive_loss_with_reg(
                feats, labels, original_feats, temperature)

            total_loss += loss.item()
            total_cont_loss += cont_loss.item()
            total_reg_loss += reg_loss.item()

    return (total_loss / len(val_loader),
            total_cont_loss / len(val_loader),
            total_reg_loss / len(val_loader))

# ====================================================
# ✅ 8. Training loop
# ====================================================
os.makedirs("checkpoints", exist_ok=True)
best_val_loss = float('inf')

print(f"\n🎯 Training Strategy for Generalization:")
print(f"  - Light fine-tuning (LR: {lr})")
print(f"  - Regularization to preserve CLIP features")
print(f"  - Strong augmentation for robustness")
print(f"  - Fewer epochs to prevent overfitting")
print(f"  - No projection head (keep 512-dim embeddings)\n")

for epoch in range(epochs):
    vision_encoder.train()
    running_loss = 0.0
    running_cont_loss = 0.0
    running_reg_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Get features from both models
        feats = vision_encoder(images)
        with torch.no_grad():
            original_feats = original_model(images)

        loss, cont_loss, reg_loss = supervised_contrastive_loss_with_reg(
            feats, labels, original_feats, temperature, reg_lambda=0.1)
        loss = loss / accum_steps

        loss.backward()

        # Gradient accumulation
        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps
        running_cont_loss += cont_loss.item()
        running_reg_loss += reg_loss.item()
        pbar.set_postfix({
            "Loss": running_loss / (step + 1),
            "Cont": running_cont_loss / (step + 1),
            "Reg": running_reg_loss / (step + 1)
        })

    # Handle remaining gradients
    if (step + 1) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    train_loss = running_loss / len(train_loader)
    train_cont = running_cont_loss / len(train_loader)
    train_reg = running_reg_loss / len(train_loader)

    # Validation
    val_loss, val_cont, val_reg = validate(vision_encoder, original_model, val_loader, device)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(f"✅ Epoch {epoch+1}/{epochs}")
    print(f"   Train - Loss: {train_loss:.4f} | Cont: {train_cont:.4f} | Reg: {train_reg:.4f}")
    print(f"   Val   - Loss: {val_loss:.4f} | Cont: {val_cont:.4f} | Reg: {val_reg:.4f} | LR: {current_lr:.6f}")

    # Save checkpoint
    checkpoint = {
        "epoch": epoch,
        "vision_encoder": vision_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, f"checkpoints/supcon_clip_generalized_epoch_{epoch+1}.pt")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(checkpoint, "checkpoints/supcon_clip_generalized_best.pt")
        print(f"   🌟 New best model saved!")

print(f"\n🎉 Training completed! Best validation loss: {best_val_loss:.4f}")
print(f"\n💡 This model should generalize better to unseen plants/diseases!")
print(f"   Test it on diverse images before using in production.")
