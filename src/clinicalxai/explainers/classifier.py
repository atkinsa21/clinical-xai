"""
Classifier explainer for ONNX binary classifiers
"""

from __future__ import annotations
from functools import cached_property

import numpy as np
import pandas as pd
import shap

from clinicalxai.explainers.base import BaseExplainer
from clinicalxai.model import OnnxModel

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve as sk_roc_curve


class ClassifierExplainer(BaseExplainer):
    def __init__(
        self,
        model: OnnxModel,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        labels: list[str] | None = None,
        protected_features: list[str] | None = None,
        background: pd.DataFrame | None = None,
    ) -> None:
        if len(X) != len(y):
            raise ValueError("Length of X and y must match.")
        self.model = model
        self.X = X
        self.y = y
        self.labels = labels if labels is not None else ["class_0", "class_1"]
        self.protected_features = (
            protected_features if protected_features is not None else []
        )
        self._background = background

    @cached_property
    def predictions(self) -> np.ndarray:
        return self.model.predict(self.X)

    @cached_property
    def shap_values(self) -> shap.Explanation:
        background = (
            self._background
            if self._background is not None
            else shap.sample(self.X, 50, random_state=0)
        )
        explainer = shap.Explainer(self.model.predict_proba, background)
        return explainer(self.X)

    @cached_property
    def metrics(self) -> dict[str, float]:
        y_pred = self.predictions
        y_proba = self._positive_class_probabilities
        return {
            "accuracy": accuracy_score(self.y, y_pred),
            "precision": precision_score(self.y, y_pred),
            "recall": recall_score(self.y, y_pred),
            "f1_score": f1_score(self.y, y_pred),
            "roc_auc": roc_auc_score(self.y, y_proba),
        }

    @cached_property
    def confusion_matrix(self) -> np.ndarray:
        return sk_confusion_matrix(self.y, self.predictions)

    @cached_property
    def roc_curve(self) -> dict[str, np.ndarray]:
        y_proba = self._positive_class_probabilities
        fpr, tpr, thresholds = sk_roc_curve(self.y, y_proba)
        return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    @cached_property
    def _positive_class_probabilities(self) -> np.ndarray:
        """
        Get the predicted probabilities for the positive class.
        """
        return self.model.predict_proba(self.X)[:, 1]
