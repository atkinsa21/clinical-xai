import base64
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

import clinicalxai.plots as plots


@pytest.fixture(scope="module")
def shap_explanation(classifier_explainer):
    shap_values = classifier_explainer.shap_values
    X = classifier_explainer.X
    return shap_values, X


def test_default_patient_index_returns_first_positive():
    predictions = np.array([0, 0, 1, 0, 1])
    index = plots.default_patient_index(predictions)
    assert index == 2


def test_default_patient_index_returns_middle_if_no_positive():
    predictions = np.array([0, 0, 0, 0])
    index = plots.default_patient_index(predictions)
    assert index == 2  # Middle index for length 4 is 2 (0-based)


def test_default_patient_index_with_different_positive_class():
    predictions = np.array([0, 2, 1, 2, 0])
    index = plots.default_patient_index(predictions, positive_class=2)
    assert index == 1


@pytest.mark.slow(reason="Using SHAP fixture can be time-consuming.")
def test_top_features_returns_indices_and_descending_values(shap_explanation):
    shap_values, _ = shap_explanation
    indices, values = plots.top_features_by_mean_abs_shap(
        shap_values, positive_class=1, top_n=5
    )

    assert isinstance(indices, list)
    assert isinstance(values, list)
    assert len(indices) == 5
    assert len(values) == 5
    assert all(isinstance(i, int) for i in indices)
    assert all(isinstance(v, float) for v in values)
    assert values == sorted(values, reverse=True)


@pytest.mark.slow(reason="Using SHAP fixture can be time-consuming.")
def test_top_features_top_n_stops_at_available_features(shap_explanation):
    shap_values, _ = shap_explanation
    total_features = shap_values.values.shape[1]
    indices, values = plots.top_features_by_mean_abs_shap(
        shap_values, positive_class=1, top_n=total_features + 10
    )

    assert len(indices) == total_features
    assert len(values) == total_features


@pytest.mark.slow(reason="Plot generation can be time-consuming.")
def test_shap_bar_html_returns_html_div(shap_explanation):
    shap_values, _ = shap_explanation
    html = plots.shap_bar_html(shap_values, positive_class=1, top_n=5)
    top_feature = plots.top_features_by_mean_abs_shap(
        shap_values, positive_class=1, top_n=5
    )[0][0]  # Get index of top feature

    assert isinstance(html, str)
    assert html.startswith("<div")
    assert html.endswith("</div>")
    assert "plotly" in html.lower()
    assert (
        shap_values.feature_names[top_feature] in html
    )  # Check that feature names are included in the HTML


@pytest.mark.slow(reason="Plot generation can be time-consuming.")
def test_shap_bar_html_does_not_include_plotlyjs_by_default(shap_explanation):
    shap_values, _ = shap_explanation
    html = plots.shap_bar_html(shap_values, positive_class=1, top_n=5)

    assert "plotly.js" not in html.lower()


def test_normalize_feature_values_scales_to_0_1():
    values = np.array([10, 20, 30])
    normalized = plots._normalize_feature_values(values)

    assert np.all(normalized >= 0) and np.all(normalized <= 1)
    np.testing.assert_allclose(normalized, [0.0, 0.5, 1.0])


def test_normalize_feature_values_with_constant_values_returns_half():
    values = np.array([5, 5, 5])
    normalized = plots._normalize_feature_values(values)

    assert np.all(normalized == 0.5)


@pytest.mark.slow(reason="Plot generation can be time-consuming.")
def test_shap_beeswarm_html_returns_html_div(shap_explanation):
    shap_values, X = shap_explanation
    html = plots.shap_beeswarm_html(shap_values, X, positive_class=1, top_n=5)
    expected_title = "Top 5 feature SHAP distribution for positive class"

    assert isinstance(html, str)
    assert html.startswith("<div>")
    assert html.endswith("</div>")
    assert "plotly" in html.lower()
    assert expected_title in html  # Check that the title is included in the HTML


def test_shap_beeswarm_html_subsamples_when_oversize(shap_explanation):
    shap_values, X = shap_explanation
    html = plots.shap_beeswarm_html(
        shap_values, X, positive_class=1, top_n=5, max_display_samples=10
    )
    assert len(html) < 200_000


@pytest.mark.slow(reason="Plot generation can be time-consuming.")
def test_shap_waterfall_html_returns_html_div(shap_explanation):
    shap_values, _ = shap_explanation
    html = plots.shap_waterfall_html(shap_values, patient_index=0, positive_class=1)

    assert isinstance(html, str)
    assert html.startswith("<div>")
    assert html.endswith("</div>")
    assert "plotly" in html.lower()
    assert "Patient at Index 0" in html  # title includes the patient index


@pytest.mark.slow(reason="Plot generation can be time-consuming.")
def test_shap_waterfall_html_includes_other_features_when_top_n_is_less_than_feature_count(
    shap_explanation,
):
    shap_values, _ = shap_explanation
    html = plots.shap_waterfall_html(
        shap_values, patient_index=0, positive_class=1, top_n=5
    )

    assert (
        "Other Features" in html
    )  # Check that "Other Features" category is included in the HTML


@pytest.mark.slow(reason="Plot generation can be time-consuming.")
def test_shap_waterfall_html_does_not_include_other_features_when_top_n_exceeds_feature_count(
    shap_explanation,
):
    shap_values, _ = shap_explanation
    total_features = shap_values.values.shape[1]
    html = plots.shap_waterfall_html(
        shap_values, patient_index=0, positive_class=1, top_n=total_features + 10
    )

    assert (
        "Other Features" not in html
    )  # "Other Features" should not be included when top_n exceeds total features


@pytest.mark.slow(reason="Plot generation can be time-consuming.")
def test_shap_waterfall_html_raises_value_error_for_negative_patient_index(
    shap_explanation,
):
    shap_values, _ = shap_explanation

    with pytest.raises(IndexError, match="out of bounds"):
        plots.shap_waterfall_html(
            shap_values, patient_index=-1, positive_class=1
        )  # Negative index


@pytest.mark.slow(reason="Plot generation can be time-consuming.")
def test_shap_waterfall_html_raises_value_error_for_out_of_bounds_patient_index(
    shap_explanation,
):
    shap_values, _ = shap_explanation
    num_samples = shap_values.values.shape[0]

    with pytest.raises(IndexError, match="out of bounds"):
        plots.shap_waterfall_html(
            shap_values, patient_index=num_samples, positive_class=1
        )  # Index equal to number of samples is out of bounds


def test_confusion_matrix_png_returns_base64_encoded_png():
    cm = np.array([[5, 2], [1, 7]])
    labels = ["Negative", "Positive"]
    png_str = plots.confusion_matrix_png(cm, labels)

    assert isinstance(png_str, str)
    assert png_str.startswith("data:image/png;base64,")

    # Try to decode the base64 string to ensure it's valid
    base64_data = png_str.split(",")[1]
    try:
        decoded_data = base64.b64decode(base64_data)
        assert len(decoded_data) > 0  # Check that we got some data back
        assert decoded_data[:8] == b"\x89PNG\r\n\x1a\n"  # Check PNG file signature
    except Exception as e:
        pytest.fail(f"Returned string is not valid base64-encoded PNG: {e}")


def test_roc_curve_png_returns_base64_encoded_png():
    fpr = np.array([0.0, 0.1, 0.4, 1.0])
    tpr = np.array([0.0, 0.6, 0.8, 1.0])
    auc = 0.85
    png_str = plots.roc_curve_png(fpr, tpr, auc)

    assert isinstance(png_str, str)
    assert png_str.startswith("data:image/png;base64,")

    # Try to decode the base64 string to ensure it's valid
    base64_data = png_str.split(",")[1]
    try:
        decoded_data = base64.b64decode(base64_data)
        assert len(decoded_data) > 0  # Check that we got some data back
        assert decoded_data[:8] == b"\x89PNG\r\n\x1a\n"  # Check PNG file signature
    except Exception as e:
        pytest.fail(f"Returned string is not valid base64-encoded PNG: {e}")


def test_mpl_fig_to_data_uri_closes_fig():
    fig = plt.figure()
    with patch("clinicalxai.plots.plt.close", wraps=plt.close) as mock_close:
        uri = plots.mpl_fig_to_data_uri(fig)
        mock_close.assert_called_once_with(fig)
    assert uri.startswith("data:image/png;base64,")


def test_mpl_fig_to_data_uri_closes_fig_on_savefig_error():
    fig = plt.figure()
    with (
        patch.object(fig, "savefig", side_effect=RuntimeError("boom")),
        patch("clinicalxai.plots.plt.close", wraps=plt.close) as mock_close,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            plots.mpl_fig_to_data_uri(fig)
        mock_close.assert_called_once_with(fig)


def test_get_plotlyjs_inline_script_returns_non_empty_string():
    script = plots.get_plotlyjs_inline_script()
    assert isinstance(script, str)
    assert len(script) > 1000
