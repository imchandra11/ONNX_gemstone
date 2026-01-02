"""
Gemstone DataModule
Lightning DataModule with preprocessing pipeline for gemstone price prediction
"""
import joblib
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import GemstoneDataset


class Preprocessor:
    """Fit/transform helper that keeps categorical levels consistent."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.cat_levels: dict[str, list[str]] = {}
        self.feature_names: list[str] = []

    def fit(self, df: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]):
        """Fit preprocessor on training data."""
        for col in categorical_cols:
            uniq = sorted(df[col].dropna().unique().tolist())
            self.cat_levels[col] = uniq + ["Unknown"]

        # Build feature ordering
        feature_names: list[str] = []
        for col in categorical_cols:
            for level in self.cat_levels[col]:
                feature_names.append(f"{col}_{level}")
        feature_names.extend(numeric_cols)
        self.feature_names = feature_names

        self.scaler.fit(df[numeric_cols])
        return self

    def transform(self, df: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        df = df.copy()
        for col in categorical_cols:
            allowed = set(self.cat_levels[col])
            df[col] = df[col].apply(lambda v: v if v in allowed else "Unknown")
            df[col] = pd.Categorical(df[col], categories=self.cat_levels[col])

        cat_df = pd.get_dummies(df[categorical_cols], prefix_sep="_", dummy_na=False)
        # Ensure all expected categorical columns exist
        for col in self.feature_names:
            if col.startswith(tuple(categorical_cols)) and col not in cat_df.columns:
                cat_df[col] = 0

        cat_df = cat_df[[c for c in self.feature_names if c in cat_df.columns]]

        num_scaled = self.scaler.transform(df[numeric_cols])
        features = np.hstack([cat_df.values, num_scaled])
        # Align final ordering
        ordered = pd.DataFrame(features, columns=self.feature_names)
        return ordered.astype(np.float32).values

    def save(self, path: Path):
        """Save preprocessor to disk."""
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
        """Load preprocessor from disk."""
        payload = joblib.load(path)
        obj = cls()
        obj.cat_levels = payload["cat_levels"]
        obj.feature_names = payload["feature_names"]
        obj.scaler.mean_ = np.array(payload["scaler_mean"])
        obj.scaler.scale_ = np.array(payload["scaler_scale"])
        return obj


class GemstoneDataModule(L.LightningDataModule):
    """Lightning DataModule for gemstone price prediction."""

    def __init__(
        self,
        csv_path: str = "gemstone.csv",
        batch_size: int = 256,
        num_workers: int = 0,
        val_split: float = 0.2,
        random_seed: int = 42,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        target_col: str = "price",
        save_preprocessor: bool = True,
        preprocessor_path: Optional[str] = None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.random_seed = random_seed
        self.target_col = target_col
        self.save_preprocessor = save_preprocessor
        self.preprocessor_path = preprocessor_path or "preprocessor.joblib"

        # Default columns if not provided
        self.categorical_cols = categorical_cols or ["cut", "color", "clarity"]
        self.numeric_cols = numeric_cols or ["carat", "depth", "table", "x", "y", "z"]

        self.preprocessor = Preprocessor()
        self.train_dataset = None
        self.val_dataset = None
        self._input_dim = None

    @property
    def input_dim(self) -> int:
        """Return input dimension for model initialization."""
        if self._input_dim is None:
            raise ValueError("DataModule must be set up first. Call setup() before accessing input_dim.")
        return self._input_dim

    def setup(self, stage: Optional[str] = None):
        """Load and preprocess data."""
        if stage == "fit" or stage is None:
            # Load data
            df = pd.read_csv(self.csv_path)
            # Drop id column if present
            if "id" in df.columns:
                df = df.drop(columns=["id"])
            df = df.dropna(subset=[self.target_col])
            features_df = df.drop(columns=[self.target_col])
            targets = df[self.target_col].astype(np.float32).values

            # Fit preprocessor on full data
            self.preprocessor.fit(features_df, self.categorical_cols, self.numeric_cols)
            X = self.preprocessor.transform(features_df, self.categorical_cols, self.numeric_cols)

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, targets, test_size=self.val_split, random_state=self.random_seed
            )

            self._input_dim = X.shape[1]

            # Create datasets
            self.train_dataset = GemstoneDataset(X_train, y_train)
            self.val_dataset = GemstoneDataset(X_val, y_val)

            # Save preprocessor if requested
            if self.save_preprocessor:
                preprocessor_path = Path(self.preprocessor_path)
                self.preprocessor.save(preprocessor_path)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Return test dataloader (uses validation set)."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
        )

