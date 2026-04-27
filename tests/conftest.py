from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from clinicalxai.datasets import load_diabetes_dataset
from clinicalxai.model import OnnxModel
from clinicalxai.explainers.classifier import ClassifierExplainer


@dataclass(frozen=True)
class OnnxFixture:
    model_path: Path
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@pytest.fixture(scope="session")
def classifier_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = load_diabetes_dataset()
    X_train, y_train = X_train.iloc[:500], y_train.iloc[:500]
    X_test, y_test = X_test.iloc[:50], y_test.iloc[:50]
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def onnx_binary_classifier(
    classifier_data, tmp_path_factory: pytest.TempPathFactory
) -> OnnxFixture:
    X_train, X_test, y_train, y_test = classifier_data
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    initial_types = [("input", FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_sklearn(
        clf, initial_types=initial_types, options={id(clf): {"zipmap": False}}
    )

    model_path = tmp_path_factory.mktemp("onnx") / "clf.onnx"
    model_path.write_bytes(onnx_model.SerializeToString())

    return OnnxFixture(
        model_path=model_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


@pytest.fixture(scope="session")
def onnx_model(onnx_binary_classifier) -> tuple[OnnxModel, OnnxFixture]:
    fx = onnx_binary_classifier
    return OnnxModel(fx.model_path), fx


@pytest.fixture(scope="session")
def classifier_explainer(onnx_model) -> ClassifierExplainer:
    model, fx = onnx_model
    return ClassifierExplainer(model, fx.X_test, fx.y_test)
