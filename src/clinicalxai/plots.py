"""
Plotting functions for generating visualizations in the XAI report.

Plotly `<div>` fragments for SHAP summary and prediction plots.
matplotlib and base64 `data:` URIs for confusion matrix and ROC curve.
"""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import shap

from plotly.offline import get_plotlyjs

SHAP_BLUE = "#1E88E5"
SHAP_RED = "#FF0051"
SHAP_COLORSCALE = [[0.0, SHAP_BLUE], [1.0, SHAP_RED]]


def default_patient_index(predictions: np.ndarray, positive_class: int = 1) -> int:
    """
    Default patient index for individual explanations. Uses first index
    where prediction is positive, or middle index if no positive predictions.

    Parameters
    ----------
    predictions : np.ndarray
        The predicted class labels for the dataset.
    positive_class : int, optional
        The class label considered as positive, by default 1.

    Returns
    -------
    int
        The index of the patient to use for individual explanations.
    """
    positive_indices = np.where(predictions == positive_class)[0]
    if len(positive_indices) > 0:
        return positive_indices[0]
    return len(predictions) // 2


def top_features_by_mean_abs_shap(
    shap_values: shap.Explanation, positive_class: int = 1, top_n: int = 10
) -> tuple[list[int], list[float]]:
    """
    Returns (indices, mean_abs_shap_values) of the top features
    for the positive class (descending order).

    Parameters
    ----------
    shap_values : shap.Explanation
        The SHAP values for the dataset.
    positive_class : int, optional
        The class label considered as positive, by default 1.
    top_n : int, optional
        The number of top features to return, by default 10.

    Returns
    -------
    tuple[list[int], list[float]]
        A tuple containing a list of feature indices and their corresponding mean absolute SHAP values.
    """
    mean_abs_shap = np.abs(shap_values.values[:, :, positive_class]).mean(axis=0)
    feature_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
    return feature_indices.tolist(), mean_abs_shap[feature_indices].tolist()


def shap_bar_html(
    shap_values: shap.Explanation, positive_class: int = 1, top_n: int = 10
) -> str:
    """
    Generate HTML for a SHAP summary bar plot of the top features for the positive class.
    Plotly <div> fragment (full_html=False, include_plotlyjs=False) for embedding in the report.

    Parameters
    ----------
    shap_values : shap.Explanation
        The SHAP values for the dataset.
    positive_class : int, optional
        The class label considered as positive, by default 1.
    top_n : int, optional
        The number of top features to display, by default 10.

    Returns
    -------
    str
        An HTML string containing the Plotly bar plot for the top SHAP features.
    """
    top_indices, mean_abs_shap = top_features_by_mean_abs_shap(
        shap_values, positive_class, top_n
    )
    fig = go.Figure(
        go.Bar(
            x=mean_abs_shap,
            y=[shap_values.feature_names[idx] for idx in top_indices],
            orientation="h",
            marker_color="rgba(255, 0, 81, 0.85)",  # SHAP_RED with some transparency
            text=[f"+{val:.3f}" for val in mean_abs_shap],
            textposition="outside",
            cliponaxis=False,  # Allow text to overflow axis limits
        )
    )
    fig.update_layout(
        title=f"Top {top_n} SHAP Features for Positive Class",
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),  # Reverse y-axis for descending order
        margin=dict(l=100, r=20, t=50, b=50),
        height=400,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def _normalize_feature_values(feature_values: np.ndarray) -> np.ndarray:
    """
    Normalize feature values to [0, 1] range for color mapping in the beeswarm plot.
    Handles constant features by returning zeros.

    Parameters
    ----------
    feature_values : np.ndarray
        The raw feature values to normalize.

    Returns
    -------
    np.ndarray
        The normalized feature values in the range [0, 1].
    """
    min_val = np.min(feature_values)
    max_val = np.max(feature_values)
    if max_val > min_val:
        norm_values = (feature_values - min_val) / (max_val - min_val)
    else:
        norm_values = np.full_like(
            feature_values, 0.5, dtype=np.float64
        )  # Constant features get a neutral color
    return norm_values


def shap_beeswarm_html(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    positive_class: int = 1,
    top_n: int = 10,
    max_display_samples=2000,
) -> str:
    """
    Generate HTML for a SHAP summary beeswarm plot for the positive class.
    Plotly <div> fragment (full_html=False, include_plotlyjs=False) for embedding in the report.

    Parameters
    ----------
    shap_values : shap.Explanation
        The SHAP values for the dataset.
    X : pd.DataFrame
        The input features corresponding to the SHAP values, used for coloring the points.
    positive_class : int, optional
        The class label considered as positive, by default 1.
    top_n : int, optional
        The number of top features to display, by default 10.
    max_display_samples : int, optional
        The maximum number of samples to display in the beeswarm plot for performance reasons, by default 2000.

    Returns
    -------
    str
        An HTML string containing the Plotly beeswarm plot for the top SHAP features.
    """
    top_indices, _ = top_features_by_mean_abs_shap(shap_values, positive_class, top_n)
    feature_names = np.asarray(shap_values.feature_names)[top_indices]
    generator = np.random.default_rng(0)
    sample_n = shap_values.values.shape[0]
    if sample_n > max_display_samples:
        generator = np.random.default_rng(0)
        sample_idx = generator.choice(sample_n, size=max_display_samples, replace=False)
    else:
        sample_idx = np.arange(sample_n)

    fig = go.Figure()
    for row, feature_idx in enumerate(top_indices):
        raw_values = np.round(X.iloc[sample_idx, feature_idx].to_numpy(), 4)
        fig.add_trace(
            go.Scatter(
                x=np.round(
                    shap_values.values[sample_idx, feature_idx, positive_class], 4
                ),
                y=np.round(
                    row + generator.uniform(-0.3, 0.3, size=len(sample_idx)), 4
                ),  # Jitter y-axis for beeswarm effect
                mode="markers",
                marker=dict(
                    size=6,
                    opacity=0.6,
                    color=_normalize_feature_values(raw_values),
                    colorscale=SHAP_COLORSCALE,
                    cmin=0,
                    cmax=1,
                    showscale=(row == 0),
                    colorbar=dict(
                        title="Feature Value", tickvals=[0, 1], ticktext=["Low", "High"]
                    ),
                ),
                customdata=raw_values,
                hovertemplate=(
                    f"<b>{feature_names[row]}</b><br>"
                    + "SHAP: %{x:.3f}<br>Value: %{customdata}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        title=f"Top {top_n} feature SHAP distribution for positive class",
        xaxis_title="SHAP Value",
        yaxis=dict(
            tickvals=list(range(len(top_indices))),
            ticktext=feature_names,
            autorange="reversed",
        ),
        height=max(400, 30 * len(top_indices) + 100),
        margin=dict(l=120, r=20, t=50, b=50),
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def shap_waterfall_html(
    shap_values: shap.Explanation,
    patient_index: int,
    positive_class: int = 1,
    top_n: int = 10,
) -> str:
    """
    Generate HTML for a SHAP waterfall plot for an individual patient.

    Parameters
    ----------
    shap_values : shap.Explanation
        The SHAP values for the dataset.
    patient_index : int
        The index of the patient to explain.
    positive_class : int, optional
        The class label considered as positive, by default 1.
    top_n : int, optional
        The number of top features to display in the waterfall plot, by default 10.

    Returns
    -------
    str
        An HTML string containing the Plotly waterfall plot for the specified patient.

    Raises
    ------
    IndexError
        If patient_index is out of bounds for the number of samples in shap_values.
    """
    if patient_index < 0 or patient_index >= shap_values.values.shape[0]:
        raise IndexError(
            f"patient_index {patient_index} is out of bounds for number of samples {shap_values.values.shape[0]}"
        )

    contributions = shap_values.values[patient_index, :, positive_class]
    patient_base = shap_values.base_values[patient_index, positive_class]
    fx_value = contributions.sum() + patient_base
    feature_names = np.asarray(shap_values.feature_names)
    order = np.argsort(np.abs(contributions))[::-1]
    top_indices, rest = order[:top_n], order[top_n:]

    fig = go.Figure(
        go.Waterfall(
            measure=["absolute"]
            + ["relative"] * (len(top_indices) + (1 if rest.size > 0 else 0)),
            x=[patient_base]
            + contributions[top_indices].tolist()
            + ([contributions[rest].sum()] if rest.size > 0 else []),
            y=["E[f(x)]"]
            + feature_names[top_indices].tolist()
            + ([f"{rest.size} Other Features"] if rest.size > 0 else []),
            orientation="h",
            increasing=dict(marker=dict(color=SHAP_RED)),
            decreasing=dict(marker=dict(color=SHAP_BLUE)),
            totals=dict(marker=dict(color="#7f7f7f")),
            connector=dict(line=dict(color="rgba(0,0,0,0.2)")),
        )
    )
    fig.update_layout(
        title=f"SHAP Waterfall for Patient at Index {patient_index}",
        xaxis_title="Cumulative SHAP Contribution f(x)",
        yaxis_title="Feature",
        margin=dict(l=50, r=20, t=50, b=50),
        height=400,
    )
    fig.add_annotation(
        x=patient_base,
        y=-0.5,
        xref="x",
        yref="y",
        text=f"E[f(x)] = {patient_base:.3f}",
        showarrow=False,
        xanchor="center",
    )
    fig.add_annotation(
        x=fx_value,
        y=len(top_indices) + (1 if rest.size > 0 else 0) - 0.5,
        xref="x",
        yref="y",
        text=f"f(x) = {fx_value:.3f}",
        showarrow=False,
        xanchor="center",
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def confusion_matrix_png(cm: np.ndarray, labels: list[str]) -> str:
    """
    Generate a PNG image of the confusion matrix as a base64-encoded
    data URI for matplotlib heatmap.

    Parameters
    ----------
    cm : np.ndarray
        The confusion matrix values.
    labels : list[str]
        The class labels corresponding to the confusion matrix axes.

    Returns
    -------
    str
        A base64-encoded PNG data URI representing the confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return mpl_fig_to_data_uri(fig)


def roc_curve_png(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> str:
    """
    Generate a PNG image of the ROC curve as a base64-encoded data URI
    for matplotlib plot.

    Parameters
    ----------
    fpr : np.ndarray
        The false positive rates for the ROC curve.
    tpr : np.ndarray
        The true positive rates for the ROC curve.
    auc : float
        The area under the ROC curve (AUC) value.

    Returns
    -------
    str
        A base64-encoded PNG data URI representing the ROC curve plot.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.lineplot(x=fpr, y=tpr, ax=ax, label=f"AUC = {auc:.2f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    plt.tight_layout()
    return mpl_fig_to_data_uri(fig)


def mpl_fig_to_data_uri(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to a base64-encoded PNG data URI.
    `plt.close(fig)` in a finally block to ensure resources are released
    even if encoding fails.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to convert.

    Returns
    -------
    str
        A base64-encoded PNG data URI representing the input matplotlib figure.
    """
    try:
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        img_data = buffer.read()
        return f"data:image/png;base64,{base64.b64encode(img_data).decode('utf-8')}"
    finally:
        plt.close(fig)


def get_plotlyjs_inline_script() -> str:
    """
    Full Plotly.js script for inline embedding in the HTML report.
    Wraps `plotly.offline.get_plotlyjs()`; One call per report --
    every fragment uses `include_plotlyjs=False` to avoid duplication.

    Returns
    -------
    str
        The full Plotly.js script as a string for inline embedding in the HTML report.
    """
    return get_plotlyjs()
