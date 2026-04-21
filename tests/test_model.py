import numpy as np
import pytest

from clinicalxai.model import OnnxModel

def test_onnx_model_loading(onnx_binary_classifier):
    model = OnnxModel(onnx_binary_classifier.model_path)
    assert model is not None

def test_onnx_model_raises_on_missing_file(tmp_path):
    missing = tmp_path / "missing_model.onnx"
    with pytest.raises(FileNotFoundError):
        OnnxModel(missing)

def test_feature_names_none_does_not_raise(onnx_binary_classifier):
    fx = onnx_binary_classifier
    model = OnnxModel(fx.model_path)
    reversed_columns = fx.X_test[list(reversed(fx.X_test.columns))]

    preds = model.predict(reversed_columns)
    probs = model.predict_proba(reversed_columns)

    assert preds.shape == (len(fx.X_test),)
    assert probs.shape == (len(fx.X_test), 2)

def test_predict_returns_1d_array_of_length_n(onnx_binary_classifier):
    fx = onnx_binary_classifier
    model = OnnxModel(fx.model_path)
    preds = model.predict(fx.X_test)
    assert preds.ndim == 1
    assert len(preds) == len(fx.X_test)

def test_predict_proba_returns_2d_array(onnx_binary_classifier):
    fx = onnx_binary_classifier
    model = OnnxModel(fx.model_path)
    probs = model.predict_proba(fx.X_test)
    assert probs.shape == (len(fx.X_test), 2)


def test_predict_proba_rows_sum_to_one(onnx_binary_classifier):
    fx = onnx_binary_classifier
    model = OnnxModel(fx.model_path)
    probs = model.predict_proba(fx.X_test)
    assert probs.sum(axis=1) == pytest.approx(1.0, abs=1e-5)


def test_predict_proba_values_in_zero_one(onnx_binary_classifier):
    fx = onnx_binary_classifier
    model = OnnxModel(fx.model_path)
    probs = model.predict_proba(fx.X_test)
    assert (probs >= 0).all() and (probs <= 1).all()

def test_predict_raises_on_wrong_feature_names(onnx_binary_classifier):
    fx = onnx_binary_classifier
    model = OnnxModel(fx.model_path, feature_names=list(fx.X_train.columns))
    reversed_columns = fx.X_test[list(reversed(fx.X_test.columns))]
    with pytest.raises(ValueError):
        model.predict(reversed_columns)

def test_predict_proba_raises_on_wrong_feature_names(onnx_binary_classifier):
    fx = onnx_binary_classifier
    model = OnnxModel(fx.model_path, feature_names=list(fx.X_train.columns))
    reversed_columns = fx.X_test[list(reversed(fx.X_test.columns))]
    with pytest.raises(ValueError):
        model.predict_proba(reversed_columns)

def test_predict_runs_when_feature_names_match(onnx_binary_classifier):
    fx = onnx_binary_classifier
    model = OnnxModel(fx.model_path, feature_names=list(fx.X_train.columns))
    preds = model.predict(fx.X_test)
    probs = model.predict_proba(fx.X_test)
    assert preds.shape == (len(fx.X_test),)
    assert probs.shape == (len(fx.X_test), 2)