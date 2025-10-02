import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import os
from tqdm import tqdm # For nice progress bars
from torchvision import transforms

from google.colab import drive
drive.mount('/content/drive')

# --- 1. Setup and GPU Configuration ---
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # This line works the same in PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Reproducibility ---
# Set seed for numpy, python random, and torch for reproducibility
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

# --- 2. Data Loading and Preprocessing ---
# This function correctly shuffles images and labels together
def load_and_prepare_data(type): # type = 'train', 'valid', or 'test'
    data_x = np.load(f'./drive/MyDrive/hori_{type}_img_gap_48_33.npy', mmap_mode='r')
    data_y = np.load(f'./drive/MyDrive/hori_{type}_label_gap_48_33.npy')

    # --- IMPORTANT FIX: Shuffle data and labels together ---
    # Create an index array, shuffle it, and apply to both data and labels
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)

    data_x = data_x[indices]
    data_y = data_y[indices]

    # Keras uses (N, H, W, C), PyTorch needs (N, C, H, W)
    # We will handle this transpose in the Dataset class
    img1_data = data_x[:, 0]
    img2_data = data_x[:, 1]

    return img1_data, img2_data, data_y, indices

#train_img1, train_img2, train_y = load_and_prepare_data('train')
#valid_img1, valid_img2, valid_y = load_and_prepare_data('valid')
#test_img1, test_img2, test_y = load_and_prepare_data('test')

# PyTorch Custom Dataset
class ImagePairDataset(Dataset):
    def __init__(self, img1_arr, img2_arr, labels):
        self.img1_arr = img1_arr
        self.img2_arr = img2_arr
        self.labels = labels
        # Assuming input numpy arrays are uint8 (0-255), we'll convert to float (0-1)
        # If they are already floats, this division can be removed.
        # EfficientNet in PyTorch also has specific normalization, but to match the
        # original code, we'll stick to a simple float conversion first.
        #self.img1_arr = self.img1_arr.astype(np.float32) / 255.0
        #self.img2_arr = self.img2_arr.astype(np.float32) / 255.0
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image pairs and label
        img1 = self.img1_arr[idx]
        img2 = self.img2_arr[idx]
        label = self.labels[idx]

        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        # Convert to PyTorch tensors and permute dimensions from (H, W, C) to (C, H, W)
        img1_tensor = self.transform(torch.from_numpy(img1).permute(2, 0, 1))
        img2_tensor = self.transform(torch.from_numpy(img2).permute(2, 0, 1))
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img1_tensor, img2_tensor, label_tensor

train_img1, train_img2, train_y, train_indices = load_and_prepare_data('train')
valid_img1, valid_img2, valid_y, valid_indices = load_and_prepare_data('valid')
#test_img1, test_img2, test_y, test_indices = load_and_prepare_data('test')

# Create Datasets
BATCH_SIZE = 200
#train_dataset = ImagePairDataset(train_img1, train_img2, train_y)
#valid_dataset = ImagePairDataset(valid_img1, valid_img2, valid_y)
#test_dataset = ImagePairDataset(test_img1, test_img2, test_y)

train_dataset = ImagePairDataset(train_img1, train_img2, train_y, train_indices)
valid_dataset = ImagePairDataset(valid_img1, valid_img2, valid_y, valid_indices)
#test_dataset = ImagePairDataset(test_img1, test_img2, test_y, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
#test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# --- 3. Modified Model Architecture (PyTorch) ---
class ComboNet(nn.Module):
    def __init__(self, input_shape=(96, 96, 3)):
        super(ComboNet, self).__init__()

        # 1. Feature Extractor Network (FEN)
        # Load pre-trained EfficientNet-B0
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # The feature extractor part of efficientnet
        feature_extractor_backbone = efficientnet.features
        # The number of output features from the backbone is 1280 for B0
        num_features = 1280

        self.fen = nn.Sequential(
            feature_extractor_backbone,
            nn.AdaptiveAvgPool2d(1), # Equivalent to GlobalAveragePooling2D
            nn.Flatten(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512), # Batch Norm for 1D data (after flatten)
            nn.ReLU(inplace=True)
        )

        # 2. Classifier Head (processes the concatenated features)
        self.classifier_head = nn.Sequential(
            # Input is 512 (from img1) + 512 (from img2) = 1024
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # Final output layer. Now it outputs probabilities using Sigmoid.
            nn.Linear(512, 1),
            nn.Sigmoid() # Sigmoid activation for binary classification (0 or 1, adjacent or not adjacent)
        )

    def forward(self, img1, img2):
        # Process each image with the shared-weight FEN
        f1_feature = self.fen(img1)
        f2_feature = self.fen(img2)

        # Concatenate features along the feature dimension (dim=1)
        concatted_feature = torch.cat((f1_feature, f2_feature), dim=1)

        # Get the final classification score (now a probability)
        output = self.classifier_head(concatted_feature)

        return output

# --- 4. Training Configuration and Execution (Modified Loss Function) ---
MODEL_PATH = 'hori_48_33_EfficientNetB0.pth' # New model name

if not os.path.exists(MODEL_PATH):
    print("Model not found. Starting training...")

    model = ComboNet().to(device)

    # Loss Function: Change to BCELoss since Sigmoid is now in the model
    criterion = nn.BCELoss() # <--- Changed from BCEWithLogitsLoss

    # Optimizer (RMSprop with same parameters as Keras)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-5, alpha=0.9, eps=1e-8, momentum=0.9)
    steps_per_epoch = len(train_loader)
    gamma = 0.95 ** (steps_per_epoch / 3600.0)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Early Stopping parameters
    patience = 20
    best_val_accuracy = 0.0
    patience_counter = 0

    # Training Loop
    MAX_EPOCHS = 100
    for epoch in range(MAX_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Use tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
        for img1, img2, labels in train_pbar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            labels = labels.unsqueeze(1) # Match output shape [B, 1]

            optimizer.zero_grad()
            outputs = model(img1, img2) # Outputs are now probabilities [0, 1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * img1.size(0)

            # Calculate accuracy
            preds = (outputs >= 0.5) # Predictions based on probabilities
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_pbar.set_postfix({'loss': loss.item(), 'acc': (preds == labels).sum().item()/labels.size(0)})

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / train_total

        # Update learning rate
        scheduler.step()

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Valid]")
            for img1, img2, labels in val_pbar:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                labels = labels.unsqueeze(1)

                outputs = model(img1, img2) # Outputs are now probabilities [0, 1]
                loss = criterion(outputs, labels)

                val_loss += loss.item() * img1.size(0)
                preds = (outputs >= 0.5) # Predictions based on probabilities
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_pbar.set_postfix({'loss': loss.item(), 'acc': (preds == labels).sum().item()/labels.size(0)})

        val_loss /= len(valid_loader.dataset)
        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} | LR: {optimizer.param_groups[0]['lr']:.8f}")

        # --- Early Stopping and Model Saving ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Validation accuracy improved to {best_val_accuracy:.4f}. Saving model to {MODEL_PATH}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation accuracy did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
print("Training Done")