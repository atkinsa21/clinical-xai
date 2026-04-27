import numpy as np
import pytest
import shap
from unittest.mock import patch

from clinicalxai.explainers.classifier import ClassifierExplainer


def test_classifier_explainer_initialization(onnx_model, classifier_explainer):
    model, fx = onnx_model
    explainer = classifier_explainer

    assert explainer.model is model
    assert explainer.X is fx.X_test
    assert explainer.y is fx.y_test
    assert explainer.labels == ["class_0", "class_1"]
    assert explainer.protected_features == []


def test_classifier_explainer_raises_on_mismatched_X_y(onnx_model):
    model, fx = onnx_model
    with pytest.raises(ValueError, match="Length of X and y must match."):
        ClassifierExplainer(model, fx.X_test, fx.y_test.iloc[:5])


def test_classifier_explainer_uses_provided_labels(onnx_model):
    model, fx = onnx_model
    custom_labels = ["negative", "positive"]
    explainer = ClassifierExplainer(model, fx.X_test, fx.y_test, labels=custom_labels)

    assert explainer.labels == custom_labels


def test_predictions_returns_array_of_correct_length(onnx_model, classifier_explainer):
    _, fx = onnx_model
    explainer = classifier_explainer
    preds = explainer.predictions

    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 1
    assert len(preds) == len(fx.X_test)


def test_predictions_property_caches_results(onnx_model, classifier_explainer):
    model, fx = onnx_model
    explainer = classifier_explainer
    explainer.__dict__.pop("predictions", None)  # Clear cache if exists

    with patch.object(model, "predict", wraps=model.predict) as mock_predict:
        preds1 = explainer.predictions
        preds2 = explainer.predictions

        assert np.array_equal(preds1, preds2)
        mock_predict.assert_called_once_with(fx.X_test)


@pytest.mark.slow(reason="SHAP explainer can be slow to compute")
def test_shap_values_returns_explanation(classifier_explainer):
    shap_values = classifier_explainer.shap_values

    assert isinstance(shap_values, shap.Explanation)


@pytest.mark.slow(reason="SHAP explainer can be slow to compute")
def test_shap_values_shape_matches_input(classifier_explainer):
    shap_values = classifier_explainer.shap_values

    assert shap_values.values.shape[0] == len(classifier_explainer.X)
    assert shap_values.values.shape[1] == classifier_explainer.X.shape[1]
    assert (
        shap_values.shape[2] == 2
    )  # binary classification should have 2 output classes


@pytest.mark.slow(reason="SHAP explainer can be slow to compute")
def test_shap_values_caches_results(classifier_explainer):
    _ = classifier_explainer.shap_values  # compute once to cache

    with patch.object(
        shap.Explainer, "__call__", wraps=shap.Explainer.__call__
    ) as mock_explainer_call:
        _ = classifier_explainer.shap_values  # access again should use cache
        mock_explainer_call.assert_not_called()


@pytest.mark.slow(reason="SHAP explainer can be slow to compute")
def test_shap_values_invariant_for_positive_class(classifier_explainer):
    """
    Test that the sum of SHAP values plus base value approximates the predicted probability for the positive class.
    """
    shap_values = classifier_explainer.shap_values
    reconstructed = (
        shap_values.values[..., 1].sum(axis=1) + shap_values.base_values[..., 1]
    )
    expected = classifier_explainer.model.predict_proba(classifier_explainer.X)[:, 1]

    np.testing.assert_allclose(reconstructed, expected, rtol=0.05)


def test_metrics_returns_expected_keys(classifier_explainer):
    metrics = classifier_explainer.metrics

    expected_keys = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
    assert set(metrics.keys()) == expected_keys


def test_metrics_in_values_between_0_and_1(classifier_explainer):
    metrics = classifier_explainer.metrics

    for metric in metrics.values():
        assert 0.0 <= metric <= 1.0


def test_metrics_is_cached(classifier_explainer):
    _ = classifier_explainer.metrics  # compute once to cache

    with patch.object(
        classifier_explainer,
        "_positive_class_probabilities",
        wraps=classifier_explainer._positive_class_probabilities,
    ) as mock_proba:
        _ = (
            classifier_explainer.metrics
        )  # access again should use cache and not recompute probabilities
        mock_proba.assert_not_called()


def test_confusion_matrix_returns_correct_shape(classifier_explainer):
    cm = classifier_explainer.confusion_matrix

    assert cm.shape == (2, 2)  # binary classification should have 2x2 confusion matrix


def test_confusion_matrix_is_cached(classifier_explainer):
    _ = classifier_explainer.confusion_matrix  # compute once to cache

    with patch.object(
        classifier_explainer, "predictions", wraps=classifier_explainer.predictions
    ) as mock_predictions:
        _ = (
            classifier_explainer.confusion_matrix
        )  # access again should use cache and not recompute predictions
        mock_predictions.assert_not_called()


def test_roc_curve_returns_fpr_tpr_thresholds(classifier_explainer):
    roc_data = classifier_explainer.roc_curve

    assert "fpr" in roc_data and "tpr" in roc_data and "thresholds" in roc_data
    assert len(roc_data["fpr"]) == len(roc_data["tpr"]) == len(roc_data["thresholds"])


def test_roc_curve_is_cached(classifier_explainer):
    _ = classifier_explainer.roc_curve  # compute once to cache

    with patch.object(
        classifier_explainer,
        "_positive_class_probabilities",
        wraps=classifier_explainer._positive_class_probabilities,
    ) as mock_proba:
        _ = (
            classifier_explainer.roc_curve
        )  # access again should use cache and not recompute probabilities
        mock_proba.assert_not_called()
