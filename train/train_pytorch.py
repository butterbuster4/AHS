import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torchvision.models import efficientnet_b0

# random seed
np.random.seed(1)
torch.manual_seed(1)

# dataset class
class CustomDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        img1 = self.data_x[0][idx]
        img2 = self.data_x[1][idx]
        label = self.data_y[idx]
        return torch.tensor(img1, dtype=torch.float32), torch.tensor(img2, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# load data function
def load_data():
    # training data
    train_data_x_1 = np.load('./npy_data/artifact_train_img_gap_5.npy')
    train_data_x_2 = np.load('./npy_data/engravings_train_img_gap_5.npy')
    train_data_x_3 = np.load('./npy_data/paintings_train_img_gap_5.npy')
    print(train_data_x_1.shape)
    print(train_data_x_2.shape)
    print(train_data_x_3.shape)
    train_data_x = np.concatenate((train_data_x_1, train_data_x_2, train_data_x_3), axis=1)
    train_y_1 = np.load('./npy_data/artifact_test_img_gap_5.npyy')
    train_y_2 = np.load('./npy_data\artifact_test_img_gap_5.npy')
    train_y_3 = np.load('./npy_data\artifact_test_img_gap_5.npy')
    print(train_y_1.shape)
    print(train_y_2.shape)
    print(train_y_3.shape)
    train_y = np.concatenate((train_y_1, train_y_2, train_y_3), axis=1)

    # shuffle training data
    indices = np.arange(len(train_y))
    np.random.shuffle(indices)
    train_data_x = [train_data_x[:, 0][indices], train_data_x[:, 1][indices]]
    train_y = train_y[indices]

    # validation data
    valid_data_x_1 = np.load('./MET_Dataset/select_image/painting_valid_img_hori_55.npy')
    valid_data_x_2 = np.load('./MET_Dataset/select_image/engraving_valid_img_hori_55.npy')
    valid_data_x_3 = np.load('./MET_Dataset/select_image/artifact_valid_img_hori_55.npy')
    valid_data_x = np.concatenate((valid_data_x_1, valid_data_x_2, valid_data_x_3), axis=1)
    valid_y_1 = np.load('./MET_Dataset/select_image/painting_valid_label_hori_55.npy')
    valid_y_2 = np.load('./MET_Dataset/select_image/engraving_valid_label_hori_55.npy')
    valid_y_3 = np.load('./MET_Dataset/select_image/artifact_valid_label_hori_55.npy')
    valid_y = np.concatenate((valid_y_1, valid_y_2, valid_y_3), axis=1)

    # shuffle validation data
    indices = np.arange(len(valid_y))
    np.random.shuffle(indices)
    valid_data_x = [valid_data_x[:, 0][indices], valid_data_x[:, 1][indices]]
    valid_y = valid_y[indices]

    return train_data_x, train_y, valid_data_x, valid_y

# base model using EfficientNet for feature extraction
class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512):
        super(EfficientNetFeatureExtractor, self).__init__()
        base_model = efficientnet_b0(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-2])  # 去掉最后的全连接层
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base_model.classifier[1].in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


class ComboNet(nn.Module):
    def __init__(self, input_shape, feature_dim=512):
        super(ComboNet, self).__init__()
        self.feature_extractor = EfficientNetFeatureExtractor(feature_dim)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        f1 = self.feature_extractor(img1)
        f2 = self.feature_extractor(img2)
        combined = torch.cat([f1, f2], dim=1)
        out = self.fc(combined)
        return out

# training function
def train_model(model, train_loader, valid_loader, device, num_epochs=100):
    criterion = nn.BCELoss()  # binary cross-entropy loss for binary classification
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    # evaluate the model on validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, labels in valid_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            outputs = model(img1, img2).squeeze()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# main function to run the training
if __name__ == "__main__":
    train_data_x, train_y, valid_data_x, valid_y = load_data()

    train_dataset = CustomDataset(train_data_x, train_y)
    valid_dataset = CustomDataset(valid_data_x, valid_y)

    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=200, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComboNet(input_shape=(96, 96, 3)).to(device)

    train_model(model, train_loader, valid_loader, device)

    # save the model
    torch.save(model.state_dict(), "combo_net.pth")