import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


CATEGORICAL_COLS = ["cut", "color", "clarity"]
NUMERIC_COLS = ["carat", "depth", "table", "x", "y", "z"]
TARGET_COL = "price"
ID_COL = "id"


class Preprocessor:
    """Fit/transform helper that keeps categorical levels consistent."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.cat_levels: Dict[str, List[str]] = {}
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame):
        for col in CATEGORICAL_COLS:
            uniq = sorted(df[col].dropna().unique().tolist())
            self.cat_levels[col] = uniq + ["Unknown"]

        # Build feature ordering
        feature_names: List[str] = []
        for col in CATEGORICAL_COLS:
            for level in self.cat_levels[col]:
                feature_names.append(f"{col}_{level}")
        feature_names.extend(NUMERIC_COLS)
        self.feature_names = feature_names

        self.scaler.fit(df[NUMERIC_COLS])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for col in CATEGORICAL_COLS:
            allowed = set(self.cat_levels[col])
            df[col] = df[col].apply(lambda v: v if v in allowed else "Unknown")
            df[col] = pd.Categorical(df[col], categories=self.cat_levels[col])

        cat_df = pd.get_dummies(df[CATEGORICAL_COLS], prefix_sep="_", dummy_na=False)
        # Ensure all expected categorical columns exist
        for col in self.feature_names:
            if col.startswith(tuple(CATEGORICAL_COLS)) and col not in cat_df.columns:
                cat_df[col] = 0

        cat_df = cat_df[[c for c in self.feature_names if c in cat_df.columns]]

        num_scaled = self.scaler.transform(df[NUMERIC_COLS])
        features = np.hstack([cat_df.values, num_scaled])
        # Align final ordering
        ordered = pd.DataFrame(features, columns=self.feature_names)
        return ordered.astype(np.float32).values

    def save(self, path: Path):
        payload = {
            "cat_levels": self.cat_levels,
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        payload = joblib.load(path)
        obj = cls()
        obj.cat_levels = payload["cat_levels"]
        obj.feature_names = payload["feature_names"]
        obj.scaler.mean_ = np.array(payload["scaler_mean"])
        obj.scaler.scale_ = np.array(payload["scaler_scale"])
        return obj


class DiamondDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.x = torch.from_numpy(features).float()
        self.y = torch.from_numpy(targets).float().view(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Regressor(L.LightningModule):
    def __init__(self, input_dim: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        hidden = 128
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden // 2, 1),
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        mae = torch.mean(torch.abs(preds - y))
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TARGET_COL])
    features_df = df.drop(columns=[TARGET_COL])
    targets = df[TARGET_COL].astype(np.float32).values
    return features_df, targets


def train_model(args):
    csv_path = Path(args.csv)
    model_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    features_df, targets = load_data(csv_path)
    preproc = Preprocessor().fit(features_df)
    X = preproc.transform(features_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, targets, test_size=0.2, random_state=42
    )
    train_ds = DiamondDataset(X_train, y_train)
    val_ds = DiamondDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = Regressor(input_dim=X.shape[1], lr=args.lr)

    ckpt_cb = ModelCheckpoint(
        dirpath=model_dir,
        filename="diamond-regressor",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_cb = EarlyStopping(monitor="val_loss", mode="min", patience=10)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[ckpt_cb, early_cb],
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = ckpt_cb.best_model_path
    model = Regressor.load_from_checkpoint(best_ckpt, input_dim=X.shape[1], lr=args.lr)
    model.eval()

    # Save preprocessor
    preproc_path = model_dir / "preprocessor.joblib"
    preproc.save(preproc_path)

    # Export to ONNX
    onnx_path = model_dir / "diamond_regressor.onnx"
    dummy = torch.from_numpy(np.zeros((1, X.shape[1]), dtype=np.float32))
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["features"],
        output_names=["price"],
        dynamic_axes={"features": {0: "batch"}, "price": {0: "batch"}},
        opset_version=17,
    )

    metadata = {
        "feature_names": preproc.feature_names,
        "categorical_levels": preproc.cat_levels,
        "onnx_model": str(onnx_path),
        "preprocessor": str(preproc_path),
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training complete. ONNX saved to {onnx_path}")
    print(f"Preprocessor saved to {preproc_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train diamond price regressor.")
    parser.add_argument("--csv", type=str, default="gemstone.csv")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)

