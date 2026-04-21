"""
ONNX model -- wrapper for ONNX Runtime inference sessions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as rt


class OnnxModel:
    """Wrapper for ONNX Runtime inference sessions."""

    def __init__(
        self, model_path: str | Path, feature_names: list[str] | None = None
    ) -> None:
        self._path = Path(model_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {self._path}")
        self._session = rt.InferenceSession(str(self._path))
        self._input_name = self._session.get_inputs()[0].name
        self._feature_names = feature_names
        outputs = self._session.get_outputs()
        self._labels_output_name = outputs[0].name
        self._proba_output_name = outputs[1].name

    def _check_columns(self, X: pd.DataFrame) -> None:
        if self._feature_names is not None and list(X.columns) != self._feature_names:
            raise ValueError(
                f"Input DataFrame feature names do not match expected: {self._feature_names}"
            )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_columns(X)
        X_arr = X.to_numpy(dtype=np.float32)
        raw = self._session.run([self._labels_output_name], {self._input_name: X_arr})[
            0
        ]
        return raw.ravel()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_columns(X)
        X_arr = X.to_numpy(dtype=np.float32)
        return self._session.run([self._proba_output_name], {self._input_name: X_arr})[
            0
        ]
