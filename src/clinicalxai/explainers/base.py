"""
Base class for all explainers in the clinical-xai package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import shap


class BaseExplainer(ABC):
    """Shared interface for all explainers."""

    @property
    @abstractmethod
    def predictions(self) -> np.ndarray:
        """Model predictions for the input data."""

    @property
    @abstractmethod
    def shap_values(self) -> shap.Explanation:
        """SHAP values for the input data."""

    @property
    @abstractmethod
    def metrics(self) -> dict[str, float]:
        """Evaluation and performance metrics for the explainer."""
