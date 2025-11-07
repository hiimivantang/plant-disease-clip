import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import open_clip

# ====================================================
# ✅ 1. Configuration
# ====================================================
train_dir = "kaggle-data/train"
val_dir = "kaggle-data/valid"  # Fixed: use 'valid' not 'val'
batch_size = 128          # Adjust based on memory
accum_steps = 2           # Gradient accumulation
epochs = 10
lr = 1e-4
temperature = 0.07
embed_dim = 256           # Projection head output size

# Fixed device detection with CUDA support
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# Adjust num_workers based on platform
num_workers = 0 if device == "mps" else 4

# ====================================================
# ✅ 2. Validate data directories
# ====================================================
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation directory not found: {val_dir}")

# ====================================================
# ✅ 3. Data pipeline
# ====================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
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
# ✅ 4. Model setup
# ====================================================
print("Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="openai"
)

# Extract vision encoder only
vision_encoder = model.visual

# Projection head for contrastive embeddings
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.normalize(x, dim=-1)

proj_head = ProjectionHead(in_dim=vision_encoder.output_dim, out_dim=embed_dim)
vision_encoder = vision_encoder.to(device)
proj_head = proj_head.to(device)
print(f"Model loaded. Vision encoder output dim: {vision_encoder.output_dim}")

# ====================================================
# ✅ 5. Supervised Contrastive Loss
# ====================================================
def supervised_contrastive_loss(features, labels, temperature=0.07):
    # Normalize features
    features = F.normalize(features, dim=1)
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

    # Loss
    loss = -mean_log_prob_pos.mean()
    return loss

# ====================================================
# ✅ 6. Optimizer and Scheduler
# ====================================================
optimizer = torch.optim.AdamW(
    list(vision_encoder.parameters()) + list(proj_head.parameters()),
    lr=lr, weight_decay=1e-4
)

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# ====================================================
# ✅ 7. Validation function
# ====================================================
def validate(vision_encoder, proj_head, val_loader, device):
    vision_encoder.eval()
    proj_head.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            feats = vision_encoder(images)
            embeds = proj_head(feats)
            loss = supervised_contrastive_loss(embeds, labels, temperature)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

# ====================================================
# ✅ 8. Training loop
# ====================================================
os.makedirs("checkpoints", exist_ok=True)
best_val_loss = float('inf')

for epoch in range(epochs):
    vision_encoder.train()
    proj_head.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Fixed autocast: use float16 for cuda, float32 for mps/cpu
        use_amp = device == "cuda"
        dtype = torch.float16 if use_amp else torch.float32

        with torch.autocast(device_type=device if device != "cpu" else "cuda",
                           dtype=dtype, enabled=use_amp):
            feats = vision_encoder(images)
            embeds = proj_head(feats)
            loss = supervised_contrastive_loss(embeds, labels, temperature)
            loss = loss / accum_steps  # Scale loss for gradient accumulation

        loss.backward()

        # Gradient accumulation
        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps  # Unscale for display
        pbar.set_postfix({"Loss": running_loss / (step + 1)})

    # Handle remaining gradients if batch doesn't divide evenly
    if (step + 1) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    train_loss = running_loss / len(train_loader)

    # Validation
    val_loss = validate(vision_encoder, proj_head, val_loader, device)

    # Learning rate scheduler step
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(f"✅ Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

    # Save checkpoint
    checkpoint = {
        "epoch": epoch,
        "vision_encoder": vision_encoder.state_dict(),
        "proj_head": proj_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, f"checkpoints/supcon_clip_epoch_{epoch+1}.pt")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(checkpoint, "checkpoints/supcon_clip_best.pt")
        print(f"🌟 New best model saved with val loss: {val_loss:.4f}")

print(f"\n🎉 Training completed! Best validation loss: {best_val_loss:.4f}")
