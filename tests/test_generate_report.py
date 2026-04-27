from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from clinicalxai import generate_report as gr_module
from clinicalxai.explainers.base import BaseExplainer
from clinicalxai.explainers.classifier import ClassifierExplainer
from clinicalxai.generate_report import generate_report


@pytest.fixture
def mock_explainer() -> MagicMock:
    mock = MagicMock(spec=ClassifierExplainer)
    mock.X = pd.DataFrame({"age": [1.0, 2.0], "sex": [0.0, 1.0], "bmi": [22.0, 28.0]})
    mock.predictions = np.array([0, 1])
    mock.shap_values = MagicMock()
    mock.metrics = {"accuracy": 0.85, "roc_auc": 0.91}
    mock.confusion_matrix = np.array([[1, 0], [0, 1]])
    mock.roc_curve = {"fpr": np.array([0.0, 1.0]), "tpr": np.array([0.0, 1.0])}
    mock.labels = ["negative", "positive"]
    return mock


@pytest.fixture
def base_explainer() -> MagicMock:
    return MagicMock(spec=BaseExplainer)


@pytest.fixture
def patched_plots(monkeypatch) -> SimpleNamespace:
    mocks = {
        "shap_bar_html": MagicMock(return_value="__SHAP_BAR_HTML__"),
        "shap_beeswarm_html": MagicMock(return_value="__SHAP_BEESWARM_HTML__"),
        "shap_waterfall_html": MagicMock(return_value="__SHAP_WATERFALL_HTML__"),
        "confusion_matrix_png": MagicMock(
            return_value="data:image/png;base64,__CONFUSION_MATRIX_PNG__"
        ),
        "roc_curve_png": MagicMock(
            return_value="data:image/png;base64,__ROC_CURVE_PNG__"
        ),
        "get_plotlyjs_inline_script": MagicMock(
            return_value="__PLOTLY_JS_INLINE_SCRIPT__"
        ),
        "default_patient_index": MagicMock(return_value=7),
        "top_features_by_mean_abs_shap": MagicMock(return_value=([0, 1], [0.5, 0.3])),
        "flag_top_features": MagicMock(return_value=[]),
    }

    for name, mock in mocks.items():
        target = gr_module if name == "flag_top_features" else gr_module.plots
        monkeypatch.setattr(target, name, mock)

    return SimpleNamespace(**mocks)


@pytest.fixture(scope="module")
def rendered_html(
    classifier_explainer, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, str]:
    out = tmp_path_factory.mktemp("report") / "report.html"
    path = generate_report(classifier_explainer, out, title="Integration Test Report")
    return path, path.read_text(encoding="utf-8")


@pytest.mark.slow(reason="Full report generation can be time-consuming.")
def test_end_to_end_with_realistic_explainer(rendered_html):
    path, html = rendered_html

    assert path.exists() and path.stat().st_size > 0
    assert html.startswith("<!DOCTYPE html>")

    assert "HighChol" in html  # Check that a feature name from the dataset is included
    assert "HeartDiseaseorAttack" in html
    assert "</html>" in html
    assert "{{ title }}" not in html
    assert "{% if ethical_flags %}" not in html


def test_raises_not_implemented_error_for_unsupported_explainer(
    base_explainer, tmp_path
):
    with pytest.raises(NotImplementedError, match="ClassifierExplainer"):
        generate_report(base_explainer, tmp_path / "report.html")


@pytest.mark.parametrize(
    "rel_path, title",
    [
        ("report.html", "Default"),
        (Path("report.html"), "Default"),
        (Path("nested/dir/report.html"), "Default"),
        ("report.html", "Etude clinique XAI"),
    ],
)
def test_path_handling_and_title(
    mock_explainer, patched_plots, tmp_path, rel_path, title
):
    out_path = tmp_path / rel_path
    result_path = generate_report(mock_explainer, out_path, title=title)

    assert result_path == out_path
    assert title in out_path.read_text(encoding="utf-8")


def test_plot_functions_called_with_expected_arguments(
    mock_explainer, patched_plots, tmp_path
):
    generate_report(mock_explainer, tmp_path / "r1.html", top_n_features=5)

    assert patched_plots.shap_bar_html.call_args.kwargs["top_n"] == 5
    assert patched_plots.shap_beeswarm_html.call_args.kwargs["top_n"] == 5
    assert patched_plots.top_features_by_mean_abs_shap.call_args.kwargs["top_n"] == 5
    assert (
        patched_plots.default_patient_index.call_args.args[0]
        is mock_explainer.predictions
    )
    assert (
        patched_plots.roc_curve_png.call_args.args[2]
        == mock_explainer.metrics["roc_auc"]
    )

    for mock in (
        patched_plots.shap_bar_html,
        patched_plots.shap_beeswarm_html,
        patched_plots.shap_waterfall_html,
        patched_plots.confusion_matrix_png,
        patched_plots.roc_curve_png,
        patched_plots.get_plotlyjs_inline_script,
        patched_plots.flag_top_features,
    ):
        mock.assert_called_once()


def test_feature_names_retrieved_from_X_columns(
    mock_explainer, patched_plots, tmp_path
):
    mock_explainer.X = pd.DataFrame(
        {
            "a": [1.0],
            "b": [2.0],
            "c": [3.0],
        }
    )

    patched_plots.top_features_by_mean_abs_shap.return_value = ([2, 0], [0.9, 0.1])
    generate_report(mock_explainer, tmp_path / "r2.html")

    patched_plots.flag_top_features.assert_called_once_with(["c", "a"])


def test_rendering_includes_all_components(mock_explainer, patched_plots, tmp_path):
    path = generate_report(mock_explainer, tmp_path / "r3.html", title="Test Report")
    html = path.read_text(encoding="utf-8")

    assert "Test Report" in html
    assert patched_plots.shap_bar_html.return_value in html
    assert patched_plots.shap_beeswarm_html.return_value in html
    assert patched_plots.shap_waterfall_html.return_value in html
    assert patched_plots.confusion_matrix_png.return_value in html
    assert patched_plots.roc_curve_png.return_value in html


def test_ethical_flags_included_in_report(mock_explainer, patched_plots, tmp_path):
    patched_plots.flag_top_features.return_value = [
        {
            "feature": "age",
            "category": "sociodemographic_factors",
            "severity": "warning",
            "rationale": "Age may indicate potential ageism bias in the model's predictions.",
            "mitigation": "Investigate model performance across different age groups.",
        },
        {
            "feature": "bmi",
            "category": "proxy_variables",
            "severity": "warning",
            "rationale": "BMI may act as a proxy for socioeconomic status, which could lead to biased predictions.",
            "mitigation": "Consider removing BMI or using techniques to mitigate bias.",
        },
    ]

    path = generate_report(
        mock_explainer, tmp_path / "r4.html", title="Ethical Flags Test"
    )
    html = path.read_text(encoding="utf-8")
    assert "Ethical Flags" in html
    assert "age" in html
    assert "bmi" in html
    assert "Age may indicate potential ageism bias" in html
    assert "BMI may act as a proxy for socioeconomic status" in html


def test_metrics_included_in_report(mock_explainer, patched_plots, tmp_path):
    path = generate_report(mock_explainer, tmp_path / "r5.html", title="Metrics Test")
    html = path.read_text(encoding="utf-8")
    assert "Metrics Summary Table" in html
    assert "Accuracy" in html
    assert "ROC AUC" in html
    assert "0.85" in html
    assert "0.91" in html


def test_default_top_n_features_is_10(mock_explainer, patched_plots, tmp_path):
    generate_report(mock_explainer, tmp_path / "r6.html")
    assert patched_plots.shap_bar_html.call_args.kwargs["top_n"] == 10
    assert patched_plots.shap_beeswarm_html.call_args.kwargs["top_n"] == 10
    assert patched_plots.top_features_by_mean_abs_shap.call_args.kwargs["top_n"] == 10


def test_generated_at_and_version_info_in_report(
    mock_explainer, patched_plots, tmp_path
):
    path = generate_report(
        mock_explainer, tmp_path / "r7.html", title="Version Info Test"
    )
    html = path.read_text(encoding="utf-8")
    assert "Report generated on" in html
    assert "UTC" in html
    assert "clinicalxai Version: 0.1.0" in html
