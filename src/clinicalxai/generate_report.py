"""
Generate HTML XAI report -- pulls cached XAI results and generates an HTML report
with visualizations, explanations, and potential bias indicators.
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from clinicalxai import __version__, plots
from clinicalxai.ethical_eval import flag_top_features
from clinicalxai.explainers.base import BaseExplainer
from clinicalxai.explainers.classifier import ClassifierExplainer


def get_metrics_labels(explainer: BaseExplainer) -> dict[str, str]:
    """
    Get human-readable labels for the metrics in the explainer's metrics dict.

    Parameters
    ----------
    explainer : BaseExplainer
        The explainer instance containing the metrics.

    Returns
    -------
    dict[str, str]
        A dictionary mapping metric keys to human-readable labels.
    """
    metric_label_map = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "roc_auc": "ROC AUC",
    }
    return {
        metric: metric_label_map.get(metric, metric) for metric in explainer.metrics
    }


def to_html(
    output_path: str | Path,
    explainer: ClassifierExplainer,
    title: str,
    shap_bar: str,
    shap_beeswarm: str,
    shap_waterfall: str,
    patient_index,
    confusion_matrix: str,
    roc_curve: str,
    ethical_flags: list[dict],
) -> None:
    """
    Render the report HTML from the Jinja2 template using precomputed report components.

    Parameters
    ----------
    output_path : str | Path
        The path where the generated HTML report will be saved.
    explainer : ClassifierExplainer
        The fitted ClassifierExplainer instance containing cached results and metrics.
    title : str
        The title of the HTML report.
    shap_bar : str
        The HTML string for the SHAP bar plot visualization.
    shap_beeswarm : str
        The HTML string for the SHAP beeswarm plot visualization.
    shap_waterfall : str
        The HTML string for the SHAP waterfall plot visualization.
    patient_index : int
        The index of the patient used for the local SHAP waterfall plot.
    confusion_matrix : str
        The base64-encoded PNG string for the confusion matrix visualization.
    roc_curve : str
        The base64-encoded PNG string for the ROC curve visualization.
    ethical_flags : list[dict]
        A list of ethical flags or recommendations based on the top SHAP features.
    """
    env = Environment(
        loader=PackageLoader("clinicalxai", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rendered_html = template.render(
        plotly_js=plots.get_plotlyjs_inline_script(),
        title=title,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        clinicalxai_version=__version__,
        metrics=explainer.metrics,
        metric_labels=get_metrics_labels(explainer),
        shap_bar=shap_bar,
        shap_beeswarm=shap_beeswarm,
        shap_waterfall=shap_waterfall,
        patient_index=patient_index,
        confusion_matrix=confusion_matrix,
        roc_curve=roc_curve,
        ethical_flags=ethical_flags,
    )
    output_path.write_text(rendered_html, encoding="utf-8")
    print(f"Report generated and saved to: {output_path}")


def render_report(
    explainer: BaseExplainer,
    output_path: str | Path,
    title: str = "Clinical XAI Report",
    top_n_features: int = 10,
) -> Path:
    """
    Render a self-contained HTML report for a fitted explainer
    (v0.1.0 supports ClassifierExplainer only).
    Pulls cached predictions, SHAP values, and metrics from the
    explainer to generate visualizations, explanations, and ethical recommendations.

    Parameters
    ----------
    explainer : BaseExplainer
        A fitted explainer instance with cached results. Must be a ClassifierExplainer for v0.1.0.
    output_path : str | Path
        The path where the generated HTML report will be saved. Parent directories will be created if they do not exist.
    title : str, optional
        The title of the HTML report, by default "Clinical XAI Report".
    top_n_features : int, optional
        The number of top SHAP features to display in the report and passed to flag_top_features, by default 10.

    Returns
    -------
    Path
        The path to the generated HTML report.

    Raises
    ------
    NotImplementedError
        If the explainer type is not supported for report generation.
    """

    if not isinstance(explainer, ClassifierExplainer):
        raise NotImplementedError(
            "Report generation currently only supports ClassifierExplainer. Please fit a ClassifierExplainer and try again."
        )

    # Generate visualizations and ethical flags
    shap_bar = plots.shap_bar_html(explainer.shap_values, top_n=top_n_features)
    shap_beeswarm = plots.shap_beeswarm_html(
        explainer.shap_values, explainer.X, top_n=top_n_features
    )
    patient_index = plots.default_patient_index(explainer.predictions)
    shap_waterfall = plots.shap_waterfall_html(
        explainer.shap_values, patient_index=patient_index, positive_class=1
    )
    confusion_matrix = plots.confusion_matrix_png(
        explainer.confusion_matrix, labels=explainer.labels
    )
    roc_curve = plots.roc_curve_png(
        explainer.roc_curve["fpr"],
        explainer.roc_curve["tpr"],
        explainer.metrics["roc_auc"],
    )

    top_features, _ = plots.top_features_by_mean_abs_shap(
        explainer.shap_values, top_n=top_n_features
    )
    top_feature_names = [explainer.X.columns[i] for i in top_features]
    ethical_flags = flag_top_features(top_feature_names)

    to_html(
        output_path=output_path,
        explainer=explainer,
        title=title,
        shap_bar=shap_bar,
        shap_beeswarm=shap_beeswarm,
        shap_waterfall=shap_waterfall,
        patient_index=patient_index,
        confusion_matrix=confusion_matrix,
        roc_curve=roc_curve,
        ethical_flags=ethical_flags,
    )
    return output_path
