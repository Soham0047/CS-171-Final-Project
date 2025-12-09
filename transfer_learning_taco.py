"""
CS171 Final Project — TACO Waste Classification
Transfer Learning with ResNet18

This script uses a pretrained ResNet18 model for waste image classification.
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Scikit-learn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("All libraries imported successfully!")

# ============================================================
# DEVICE SETUP
# ============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH = "./"
DATA_DIR = "./TACO_crops"
SPLITS_DIR = "./splits_crops"

# TrashNet 6 classes (mapped from TACO)
CLASSES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
NUM_CLASSES = len(CLASSES)

# ResNet expects 224x224 images
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001  # Lower learning rate to prevent overfitting
WEIGHT_DECAY = 0.01     # L2 regularization

# ImageNet normalization values (required for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

# Load CSV splits
train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))

# Fix filepaths
def fix_filepath(fp):
    if BASE_PATH and not fp.startswith(BASE_PATH):
        return os.path.join(BASE_PATH, fp)
    return fp

train_df['filepath'] = train_df['filepath'].apply(fix_filepath)
val_df['filepath'] = val_df['filepath'].apply(fix_filepath)
test_df['filepath'] = test_df['filepath'].apply(fix_filepath)

# Create label index mapping
class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
train_df['label_idx'] = train_df['label'].map(class_to_idx)
val_df['label_idx'] = val_df['label'].map(class_to_idx)
test_df['label_idx'] = test_df['label'].map(class_to_idx)

print(f"Dataset splits loaded:")
print(f"  Train: {len(train_df):4d} samples")
print(f"  Val:   {len(val_df):4d} samples")
print(f"  Test:  {len(test_df):4d} samples")

# ============================================================
# BALANCE THE DATASET
# ============================================================
print("\n" + "="*60)
print("BALANCING DATASET")
print("="*60)

MAX_TRAIN = 600
MAX_VAL_TEST = 50

def balance(df, max_n):
    balanced = []
    for cls in CLASSES:
        subset = df[df["label"] == cls]
        if len(subset) > max_n:
            subset = subset.sample(max_n, random_state=42)
        balanced.append(subset)
    return pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)

train_df = balance(train_df, MAX_TRAIN)
val_df = balance(val_df, MAX_VAL_TEST)
test_df = balance(test_df, MAX_VAL_TEST)

print(f"\nBalanced Training set:")
print(train_df['label'].value_counts())
print(f"\nFinal dataset sizes:")
print(f"  Train: {len(train_df):4d} samples")
print(f"  Val:   {len(val_df):4d} samples")
print(f"  Test:  {len(test_df):4d} samples")

# ============================================================
# DATASET CLASS
# ============================================================
class TrashDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['filepath']).convert('RGB')
        label = row['label_idx']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================
# DATA TRANSFORMS (ImageNet normalization)
# ============================================================
# Training transform with MORE AGGRESSIVE augmentation to reduce overfitting
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # Resize larger first
    transforms.RandomCrop(IMG_SIZE),                     # Then random crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(30),                       # More rotation
    transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.8, 1.2)),  # More aggressive
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),                   # Occasionally grayscale
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Blur augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))  # Random erasing
])

# Validation/Test transform (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ============================================================
# CREATE DATA LOADERS
# ============================================================
train_dataset = TrashDataset(train_df, transform=train_transform)
val_dataset = TrashDataset(val_df, transform=test_transform)
test_dataset = TrashDataset(test_df, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")
print(f"  Test batches:  {len(test_loader)}")

# ============================================================
# TRANSFER LEARNING MODEL (ResNet18)
# ============================================================
print("\n" + "="*60)
print("BUILDING TRANSFER LEARNING MODEL (ResNet18)")
print("="*60)

class TransferLearningModel(nn.Module):
    """
    Transfer Learning using pretrained ResNet18.
    - Freeze early layers (feature extraction)
    - Replace final FC layer for our 6 classes
    - Fine-tune later layers
    """
    def __init__(self, num_classes=6, freeze_features=True):
        super(TransferLearningModel, self).__init__()

        # Load pretrained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze feature extraction layers (optional)
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False

            # Unfreeze layer3 and layer4 for better fine-tuning
            for param in self.resnet.layer3.parameters():
                param.requires_grad = True
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True

        # Replace the final fully connected layer with more regularization
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.6),                    # Higher dropout
            nn.Linear(num_features, 128),       # Smaller hidden layer
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Create model
model = TransferLearningModel(num_classes=NUM_CLASSES, freeze_features=True).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"\nModel Architecture: ResNet18 (pretrained on ImageNet)")
print(f"  Total parameters:     {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Frozen parameters:    {frozen_params:,}")

# ============================================================
# TRAINING SETUP
# ============================================================
# Compute class weights for imbalanced data
labels_np = train_df["label_idx"].values
weights = compute_class_weight("balanced", classes=np.unique(labels_np), y=labels_np)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

print("\nClass weights for loss function:")
for cls, w in zip(CLASSES, weights):
    print(f"  {cls:12s}: {w:.4f}")

# Loss function with class weights and label smoothing
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# Only optimize trainable parameters with weight decay (L2 regularization)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model, train_loader, val_loader, num_epochs=50):
    """Training loop for transfer learning model."""
    best_acc = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("\n" + "="*60)
    print("TRAINING TRANSFER LEARNING MODEL")
    print("="*60)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        # Step scheduler based on validation loss
        scheduler.step(epoch_val_loss)

        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Save best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_transfer_model.pth')
            print(f"  -> New best model saved! (Val Acc: {best_acc:.4f})")

    print(f"\nTraining complete. Best Validation Accuracy: {best_acc:.4f}")
    return train_losses, val_losses, train_accs, val_accs

# ============================================================
# TRAIN THE MODEL
# ============================================================
train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, val_loader, num_epochs=NUM_EPOCHS
)

# ============================================================
# PLOT TRAINING PROGRESS
# ============================================================
print("\n" + "="*60)
print("TRAINING PROGRESS")
print("="*60)

actual_epochs = len(train_losses)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(range(1, actual_epochs+1), train_losses, label='Train Loss')
axes[0].plot(range(1, actual_epochs+1), val_losses, label='Val Loss')
axes[0].set_title('Transfer Learning: Loss over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(range(1, actual_epochs+1), train_accs, label='Train Acc')
axes[1].plot(range(1, actual_epochs+1), val_accs, label='Val Acc')
axes[1].set_title('Transfer Learning: Accuracy over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transfer_learning_training_progress.png', dpi=150)
plt.show()

print(f"\nTraining completed after {actual_epochs} epochs")
print(f"Best Validation Accuracy: {max(val_accs):.4f}")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

# Load best model
best_model = TransferLearningModel(num_classes=NUM_CLASSES, freeze_features=True).to(device)
best_model.load_state_dict(torch.load('best_transfer_model.pth'))
best_model.eval()

# Evaluate on test set
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing Transfer Learning Model"):
        images, labels = images.to(device), labels.to(device)
        outputs = best_model(images)
        _, predicted = torch.max(outputs.data, 1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = test_correct / test_total

print(f"\n" + "="*60)
print("TRANSFER LEARNING TEST RESULTS")
print("="*60)
print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
print("\n" + "-"*60)
print("Classification Report (Transfer Learning - ResNet18):")
print("-"*60)
print(classification_report(all_labels, all_preds, target_names=CLASSES))

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
plt.title('Transfer Learning (ResNet18) Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(CLASSES))
plt.xticks(tick_marks, CLASSES, rotation=45, ha='right')
plt.yticks(tick_marks, CLASSES)

# Add counts
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('transfer_learning_confusion_matrix.png', dpi=150)
plt.show()

# Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)
print("\nPer-Class Accuracy (Transfer Learning):")
for cls, acc in zip(CLASSES, per_class_acc):
    print(f"  {cls:12s}: {acc:.4f} ({acc*100:.1f}%)")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nModel: ResNet18 (pretrained on ImageNet)")
print(f"Dataset: TACO Waste Classification (6 classes)")
print(f"Training Samples: {len(train_df)}")
print(f"Test Samples: {len(test_df)}")
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
print(f"Best Validation Accuracy: {max(val_accs)*100:.2f}%")
print(f"\nModel saved to: best_transfer_model.pth")
print(f"Training plot saved to: transfer_learning_training_progress.png")
print(f"Confusion matrix saved to: transfer_learning_confusion_matrix.png")
print("\n" + "="*60)
print("TRANSFER LEARNING COMPLETE!")
print("="*60)
