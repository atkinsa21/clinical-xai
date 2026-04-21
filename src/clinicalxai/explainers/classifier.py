"""
Classifier explainer for ONNX binary classifiers
"""
from __future__ import annotations
from functools import cached_property

import numpy as np
import pandas as pd

from clinicalxai.explainers.base import BaseExplainer
from clinicalxai.model import OnnxModel

class ClassifierExplainer(BaseExplainer):
    def __init__(
            self, 
            model: OnnxModel, 
            X: pd.DataFrame, 
            y: pd.Series, 
            *, 
            labels: list[str] | None = None, 
            protected_features: list[str] | None = None, 
            ) -> None:
        if len(X) != len(y):
            raise ValueError("Length of X and y must match.")
        self.model = model
        self.X = X
        self.y = y
        self.labels = labels if labels is not None else ["class_0", "class_1"]
        self.protected_features = protected_features if protected_features is not None else []

    @cached_property
    def predictions(self) -> np.ndarray:
        return self.model.predict(self.X)
    
    @cached_property
    def shap_values(self) -> np.ndarray:
        raise NotImplementedError("SHAP value computation not implemented yet.")
    
    @cached_property
    def metrics(self) -> dict[str, float]:
        raise NotImplementedError("Metrics computation not implemented yet.")
