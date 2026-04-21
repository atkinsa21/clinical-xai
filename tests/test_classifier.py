import numpy as np
import pytest
from unittest.mock import patch

from clinicalxai.explainers.classifier import ClassifierExplainer
from clinicalxai.model import OnnxModel

def test_classifier_explainer_initialization(onnx_binary_classifier):
    fx = onnx_binary_classifier
    model = OnnxModel(fx.model_path)
    explainer = ClassifierExplainer(model, fx.X_test, fx.y_test)

    assert explainer.model is model
    assert explainer.X is fx.X_test
    assert explainer.y is fx.y_test
    assert explainer.labels == ["class_0", "class_1"]
    assert explainer.protected_features == []

def test_classifier_explainer_raises_on_mismatched_X_y(onnx_binary_classifier):
    fx = onnx_binary_classifier
    model = OnnxModel(fx.model_path)
    with pytest.raises(ValueError, match="Length of X and y must match."):
        ClassifierExplainer(model, fx.X_test, fx.y_test.iloc[:5])

def test_classifier_explainer_uses_provided_labels(onnx_binary_classifier):
    fx = onnx_binary_classifier
    model = OnnxModel(fx.model_path)

    custom_labels = ["negative", "positive"]
    explainer = ClassifierExplainer(model, fx.X_test, fx.y_test, labels=custom_labels)
    
    assert explainer.labels == custom_labels

def test_predictions_returns_array_of_correct_length(onnx_binary_classifier):
    fx = onnx_binary_classifier
    model = OnnxModel(fx.model_path)
    explainer = ClassifierExplainer(model, fx.X_test, fx.y_test)

    preds = explainer.predictions

    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 1
    assert len(preds) == len(fx.X_test)

def test_predictions_property_caches_results(onnx_binary_classifier):
    fx = onnx_binary_classifier
    model = OnnxModel(fx.model_path)
    explainer = ClassifierExplainer(model, fx.X_test, fx.y_test)

    with patch.object(model, "predict", wraps=model.predict) as mock_predict:
        preds1 = explainer.predictions
        preds2 = explainer.predictions

        assert np.array_equal(preds1, preds2)
        mock_predict.assert_called_once_with(fx.X_test)