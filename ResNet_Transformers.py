import os
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils.class_weight import compute_class_weight

FS = 300
WINDOW_SIZE = 30 * FS  
CLASSES = ['A', 'N', 'O', '~']
BATCH_SIZE = 64
EPOCHS = 50

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)  
        
        out += identity  
        out = self.relu(out)
        
        return out

class ResNetECG(nn.Module):
    def __init__(self, input_size=WINDOW_SIZE, num_classes=4):
        super(ResNetECG, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.res_block1 = ResidualBlock(64, 128)
        self.res_block2 = ResidualBlock(128, 256)
        self.res_block3 = ResidualBlock(256, 512)

        # Fully connected layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Data Augmentation: Add Noise
def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data)) * noise_factor
    augmented_data = data + noise
    return augmented_data

# Data Preprocessing: Load and augment data
def load_data(data_dir, reference_file, max_len=WINDOW_SIZE):
    df = pd.read_csv(reference_file, header=None)
    df.columns = ['filename', 'label']
    
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    signals = []
    labels = []

    for i in range(len(df)):
        file_name = df['filename'].iloc[i]
        label = df['label_encoded'].iloc[i]
        mat_file = scipy.io.loadmat(os.path.join(data_dir, f"{file_name}.mat"))
        ecg_signal = mat_file['val'][0]

        # Preprocessing: normalize and add noise
        ecg_signal = np.nan_to_num(ecg_signal)
        ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        ecg_signal = add_noise(ecg_signal, noise_factor=0.005)

        # Pad/truncate to max_len
        padded_signal = np.zeros(max_len)
        padded_signal[:min(max_len, len(ecg_signal))] = ecg_signal[:min(max_len, len(ecg_signal))]

        signals.append(torch.tensor(padded_signal, dtype=torch.float32))
        labels.append(torch.tensor(label, dtype=torch.long))

    signals_padded = pad_sequence(signals, batch_first=True, padding_value=0.0)
    return signals_padded, torch.tensor(labels)

# Training function with K-Fold cross-validation
def train_model_with_cross_validation(X, y, model, criterion, optimizer, device):
    kf = KFold(n_splits=5)
    best_f1 = 0
    fold = 1

    for train_index, val_index in kf.split(X):
        print(f"\nTraining Fold {fold}...\n")
        fold += 1

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # DataLoader
        train_dataset = TensorDataset(X_train.unsqueeze(1), y_train)
        val_dataset = TensorDataset(X_val.unsqueeze(1), y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            f1 = f1_score(all_labels, all_preds, average='macro')
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, F1 Score: {f1:.4f}')

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), 'best_model_fold_{}.pth'.format(fold))

        print(f"\nFold {fold} F1 Score: {best_f1:.4f}\n")

def main():
    data_dir = 'training2017/'
    reference_file = 'REFERENCE-v1.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, y = load_data(data_dir, reference_file)
    print(f"Data loaded: {X.shape} samples.")

    unique_classes = np.unique(y.cpu().numpy())
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y.cpu().numpy())
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = ResNetECG().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    train_model_with_cross_validation(X, y, model, criterion, optimizer, device)

if __name__ == "__main__":
    main()
