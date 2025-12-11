import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import Dataset, DataLoader
import glob
import matplotlib.pyplot as plt
import numpy as np
import json


class EEGDataset(Dataset):
    def __init__(self, X, y):
        super(EEGDataset, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_mini_cnn(num_classes=2):
    """
    Simple CNN for spectrogram classification.
    Input: (batch, 19, 33, 7)
    Output: logits (batch, num_classes)
    """
    model = nn.Sequential(
        # Block 1
        nn.Conv2d(19, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        # Block 2
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        # Block 3
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        
        # Global average pooling
        nn.AdaptiveAvgPool2d((1, 1)),
        
        # Classifier
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(64, num_classes)
    )
    return model


def set_weights_criterion(y_train, num_classes, device):
    unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
    total_train = len(y_train)
    class_weights_dict = {label: total_train / (count + 1e-8) for label, count in zip(unique_y_train, counts_y_train)}
    weights_list = [class_weights_dict.get(i, 1.0) for i in range(num_classes)]
    weights = torch.tensor(weights_list, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    print(f'  Class weights: {weights.cpu().numpy()}')
    return criterion


def per_sample_normalization(X, eps=1e-8):
    """Normalize each sample to zero mean and unit variance"""
    new_x = X.copy()
    for i in range(new_x.shape[0]):
        m = new_x[i].mean()
        s = new_x[i].std()
        new_x[i] = (new_x[i] - m) / (s + eps)

    return new_x


def can_stratify(labels):
    """Check if stratification is possible (each class has >= 2 samples)"""
    unique, counts = np.unique(labels, return_counts=True)
    return np.all(counts >= 2)


def set_device():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def create_mask_from_patient_ids(patient_ids, selected_patient_ids):
    mask = np.isin(patient_ids, selected_patient_ids)
    return mask


def split_patient_training(patient_ids, y_all):
    unique_patients = np.unique(patient_ids)
    if unique_patients.size < 2:
        print(f'  Only {unique_patients.size} patient(s) in cluster; skipping patient-level split. Skipping cluster.')
        return None, None, None

    patient_labels = np.array([y_all[np.where(patient_ids == pid)[0][0]] for pid in unique_patients])

    # First split: train vs temp (use stratify only when possible)
    strat_1 = patient_labels if can_stratify(patient_labels) else None
    train_patient_ids, temp_patient_ids = train_test_split(
        unique_patients,
        test_size=0.30,
        stratify=strat_1,
        random_state=42
    )

    temp_labels = np.array([y_all[np.where(patient_ids == pid)[0][0]] for pid in temp_patient_ids])
    strat_2 = temp_labels if can_stratify(temp_labels) else None
    # If temp_patient_ids has fewer than 2 patients, fall back: assign all to val (or test) appropriately
    if len(temp_patient_ids) < 2:
        # Move one patient from train to val if possible, otherwise assign all temp to val
        if len(train_patient_ids) >= 1:
            # Move last train patient into val to ensure at least one patient in val/test
            moved = train_patient_ids[-1]
            train_patient_ids = train_patient_ids[:-1]
            val_patient_ids = np.array([moved])
            test_patient_ids = np.array([])
        else:
            val_patient_ids = temp_patient_ids
            test_patient_ids = np.array([])
    else:
        val_patient_ids, test_patient_ids = train_test_split(
            temp_patient_ids,
            test_size=0.50,
            stratify=strat_2,
            random_state=42
        )

    print(f'  Patient-level split:')
    print(f'    Train: {len(train_patient_ids)} patients')
    print(f'    Val:   {len(val_patient_ids)} patients')
    print(f'    Test:  {len(test_patient_ids)} patients')

    return train_patient_ids, val_patient_ids, test_patient_ids


def load_patient_data(patient_dir):
    CHANNELS = 19
    FREQ = 33
    TIME = 7
    FLATTENED_DIM = CHANNELS * FREQ * TIME
    
    X_parts, y_parts, patient_ids = [], [], []
    for xf in sorted(glob.glob(os.path.join(patient_dir, 'sub-*_X.npy'))):
        pid = os.path.basename(xf).split('_')[0]  # e.g., 'sub-001'
        Xp = np.load(xf)
        yp = np.load(os.path.join(patient_dir, f'{pid}_y.npy'))
        X_parts.append(Xp)
        y_parts.append(yp)
        patient_ids.extend([pid] * len(yp))  # associate each epoch with patient ID

    if len(X_parts) == 0:
        print(f'No data found in {patient_dir}, skipping...')
        return None, None, None
    
    # Concatenate all patients in this cluster (epochs from all patients)
    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    patient_ids = np.array(patient_ids)
    
    print(f'Loaded data shape: {X_all.shape}, labels: {np.unique(y_all, return_counts=True)}')

    # Reshape if flattened
    if X_all.ndim == 2 and X_all.shape[1] == FLATTENED_DIM:
        X_all = X_all.reshape((-1, CHANNELS, FREQ, TIME))
    
    # --- Patient-level split (avoids data leakage) ---
    # Get unique patients and their labels
    unique_patients = np.unique(patient_ids)
    if unique_patients.size < 2:
        print(f'Only {unique_patients.size} patient(s) in cluster; skipping patient-level split. Skipping cluster.')
        return None, None, None

    return X_all, y_all, patient_ids


def concatenate_patient_data(X, y, patient_ids):
    CHANNELS = 19
    FREQ = 33
    TIME = 7
    FLATTENED_DIM = CHANNELS * FREQ * TIME

    # Concatenate all patients in this cluster (epochs from all patients)
    X_all = np.concatenate(X, axis=0)
    y_all = np.concatenate(y, axis=0)
    patient_ids_con = np.array(patient_ids)

    # Reshape if flattened
    if X_all.ndim == 2 and X_all.shape[1] == FLATTENED_DIM:
        X_all = X_all.reshape((-1, CHANNELS, FREQ, TIME))

    return X_all, y_all, patient_ids_con


def test_model(model, test_loader, device):
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0

        return test_accuracy


def process_all_clusters(cluster_base_dir, output_dir):
    eps = 1e-8
    os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cluster_dirs = sorted([d for d in glob.glob(os.path.join(cluster_base_dir, 'cluster_*')) if os.path.isdir(d)])
    print(f'Found {len(cluster_dirs)} clusters: {cluster_dirs}\n')

    for cluster_dir in cluster_dirs:
        cluster_name = os.path.basename(cluster_dir)
        print(f'\n{"="*60}')
        print(f'Processing {cluster_name}')
        print(f'{"="*60}')

        # Load per-patient files from cluster
        X_all, y_all, patient_ids_con = load_patient_data(cluster_dir)

        if X_all is None and y_all is None and patient_ids_con is None:
            print(f'Skipping cluster {cluster_name} due to insufficient data.')
            continue

        train_patient_ids, val_patient_ids, test_patient_ids = split_patient_training(patient_ids_con, y_all)
        if train_patient_ids is None:
            print(f'Skipping cluster {cluster_name} because patient-level split could not be performed.')
            continue


        train_mask = create_mask_from_patient_ids(patient_ids_con, train_patient_ids)
        val_mask = create_mask_from_patient_ids(patient_ids_con, val_patient_ids)
        test_mask = create_mask_from_patient_ids(patient_ids_con, test_patient_ids)

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        print(f'  Epoch counts after concatenation:')
        print(f'    Train: {len(X_train)} epochs')
        print(f'    Val:   {len(X_val)} epochs')
        print(f'    Test:  {len(X_test)} epochs')

        X_train = per_sample_normalization(X_train, eps)
        X_val = per_sample_normalization(X_val, eps)
        X_test = per_sample_normalization(X_test, eps)

        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        test_dataset = EEGDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Build model
        num_classes = len(np.unique(y_all))
        model = build_mini_cnn(num_classes=num_classes).to(device)

        criterion = set_weights_criterion(y_train, num_classes, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        train_losses, train_accs, val_losses, val_accs, best_val_acc = train_individual_cluster(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, output_base_dir=output_dir, cluster_name=cluster_name)

        test_accuracy = test_model(model, test_loader, device)
        print(f'  Test Accuracy: {test_accuracy:.2f}%')

        # Save metrics
        metrics = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'test_accuracy': test_accuracy,
            'best_val_acc': best_val_acc
        }
        out_dir = os.path.join(output_dir, cluster_name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)




def train_individual_cluster(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, output_base_dir='', cluster_name=''):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_loss /= len(train_loader)
        train_acc = correct / total if total > 0 else 0

        train_losses.append(train_loss)
        train_accs.append(train_acc)

         # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_dir = os.path.join(output_base_dir, cluster_name)
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_dir, f'best_model_{best_val_acc:.4f}.pth'))
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')


    return train_losses, train_accs, val_losses, val_accs, best_val_acc


if __name__ == "__main__":
    cluster_base_dir = 'data/clustered_k6' #Base directory containing post-processed clustered data
    output_base_dir = 'models/cluster_models' #Where the best model checkpoints and metrics will be saved
    process_all_clusters(cluster_base_dir=cluster_base_dir, output_dir=output_base_dir)