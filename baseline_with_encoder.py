"""
FINAL BASELINE THAT WORKS WITH model.safetensors
===========================================================
"""
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import argparse
import os

from safetensors.torch import load_file

# ChannelNet imports
from channelnet.config import EEGModelConfig
from channelnet.model import ChannelNetModel


# ============================================================
# PTM Dataset (raw)
# ============================================================

class PTM_EEG_Dataset(Dataset):
    def __init__(self, eeg_path, time_low=0, time_high=450):
        loaded = torch.load(eeg_path)
        self.data = loaded["dataset"]
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.time_low = time_low
        self.time_high = time_high

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        rec = self.data[i]

        eeg = rec["eeg"].float()                     # [128, T_raw]
        eeg = eeg[:, self.time_low:self.time_high]   # [128, 450]
        eeg = eeg.unsqueeze(0).unsqueeze(0)          # [1,1,128,450]
        eeg = F.interpolate(eeg, size=(128, 440), mode="bilinear", align_corners=False)

        label = rec["label"]
        return eeg, label


# ============================================================
# PTM Splits
# ============================================================

class PTM_Split(Dataset):
    def __init__(self, base_ds, split_path, split_num, split_name):
        self.base = base_ds
        loaded = torch.load(split_path)

        idxs = loaded["splits"][split_num][split_name]

        self.indices = [
            i for i in idxs
            if 450 <= self.base.data[i]["eeg"].size(1) <= 600
        ]

        print(f"[{split_name}] Using {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


# ============================================================
# Load Stage1 Encoder
# ============================================================

def load_channelnet_encoder(folder):
    print(f"Loading ChannelNet encoder from {folder}")

    config = EEGModelConfig.from_pretrained(folder)

    encoder = ChannelNetModel(config)

    weight_path = os.path.join(folder, "model.safetensors")
    print("Loading weights:", weight_path)

    state = load_file(weight_path)
    encoder.load_state_dict(state)

    encoder.eval()
    if torch.cuda.is_available():
        encoder.cuda()

    return encoder


# ============================================================
# Extract embedding
# ============================================================

def extract_embedding(encoder, eeg):
    with torch.no_grad():
        eeg = eeg.cuda() if torch.cuda.is_available() else eeg
        emb, cls = encoder(eeg)
        emb = emb.squeeze(0).cpu().numpy()
    return emb


def build_embeddings(ds, encoder):
    X, y = [], []
    for eeg, label in ds:
        X.append(extract_embedding(encoder, eeg))
        y.append(label)
    return np.stack(X), np.array(y)


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, nclass):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, nclass),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_dataset", required=True)
    parser.add_argument("--splits_path", required=True)
    parser.add_argument("--encoder_path", required=True)
    parser.add_argument("--split_num", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    print("\nLoading PTM dataset...")
    base_ds = PTM_EEG_Dataset(args.eeg_dataset)

    train_ds = PTM_Split(base_ds, args.splits_path, args.split_num, "train")
    val_ds   = PTM_Split(base_ds, args.splits_path, args.split_num, "val")
    test_ds  = PTM_Split(base_ds, args.splits_path, args.split_num, "test")

    encoder = load_channelnet_encoder(args.encoder_path)

    print("\nExtracting embeddings...")
    X_train, y_train = build_embeddings(train_ds, encoder)
    X_val,   y_val   = build_embeddings(val_ds, encoder)
    X_test,  y_test  = build_embeddings(test_ds, encoder)

    # ---------------- Logistic Regression ----------------
    print("\n===== Logistic Regression =====")
    clf = LogisticRegression(max_iter=3000, multi_class="multinomial")
    clf.fit(X_train, y_train)
    print("Train Acc:", accuracy_score(y_train, clf.predict(X_train)))
    print("Val   Acc:", accuracy_score(y_val,   clf.predict(X_val)))
    print("Test  Acc:", accuracy_score(y_test,  clf.predict(X_test)))

    # ---------------- MLP ----------------
    print("\n===== MLP =====")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(list(zip(X_train_t, y_train_t)), batch_size=128, shuffle=True)
    val_loader   = DataLoader(list(zip(X_val_t, y_val_t)), batch_size=256)
    test_loader  = DataLoader(list(zip(X_test_t, y_test_t)), batch_size=256)

    model = MLP(X_train.shape[1], len(np.unique(y_train))).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_f = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_f(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val_t.to(device)).argmax(1).cpu()
            acc = (preds == y_val).float().mean().item()
        print(f"Epoch {ep} | Val Acc: {acc:.4f}")

    with torch.no_grad():
        preds = model(X_test_t.to(device)).argmax(1).cpu()
        test_acc = (preds == y_test).float().mean().item()

    print("\nMLP Test Acc:", test_acc)


if __name__ == "__main__":
    main()
