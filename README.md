![PyPI](https://img.shields.io/pypi/v/clinicalxai?pypiBaseUrl=https%3A%2F%2Ftest.pypi.org)
[![CI Build](https://github.com/atkinsa21/clinical-xai/actions/workflows/ci-build.yml/badge.svg?branch=main)](https://github.com/atkinsa21/clinical-xai/actions/workflows/ci-build.yml)
[![codecov](https://codecov.io/github/atkinsa21/clinical-xai/graph/badge.svg?token=YZA3V1RRBP)](https://codecov.io/github/atkinsa21/clinical-xai)

# clinical-xai
An ethical eXplainable AI (XAI) tool for clinical model analysis and reporting. This package makes machine learning model evaluation and explainability easy by preparing and exporting all performance- and feature-related metrics that are relevant for clinical data modeling.

## What is Included:
**Performance Metrics -** accuracy, precision, recall, F1, and ROC-AUC scores.

**Feature Importance Metrics -** `SHAP` values, global/local `SHAP` visualizations (beeswarm and waterfall), confusion matrix, and ROC curve. 

**Ethical Evaluation -** Working with machine learning models in high-risk settings, such as hospitals, carry significant ethical risks and moral responsibility. `clinicalxai` will evaluate the top features used in the model and will flag any that may result in unfair and biased predictions. Feature categories worth flagging include: sociodemographic factors (sex, gender, race, etc.), proxy variables (ICD diagnostic codes, insurance, number of visits, etc.), and socioeconomic factors (zipcode, income, education level, etc.)

**Static HTML Reporting -** Formats all available metrics into an easy-to-read html.

## Install
`clinicalxai` is published on TestPyPI. Some dependencies (`shap`, `onnxruntime`, `plotly`) live only on real PyPI, so you must include both indexes:
```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  clinicalxai
```
Requires Python 3.10+.
## Quick Start
Before using `clinicalxai`, you will need to convert your model (v0.1.0 - `sklearn` binary classification only) to an ONNX model equivalent. For example, to convert a logistic regression model:
```python
from sklearn.linear_model import LogisticRegression
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
initial_types = [("input", FloatTensorType([None, X_train.shape[1]]))]

onnx_model = convert_sklearn(clf,
                             initial_types=initial_types,
                             options={id(clf): {"zipmap": False}})

model_path = Path(__file__).resolve().parent / "diabetes_clf.onnx" # path to save .onnx file
model_path.write_bytes(onnx_model.SerializeToString())
```

Once the `.onnx` file is saved, you can then load your model and its attributes using `OnnxModel()` before running `ClassifierExplainer()`:
```python
from clinicalxai.model import OnnxModel
from clinicalxai.explainers.classifier import ClassifierExplainer

model = OnnxModel(model_path) # /your/path/to/<model>.onnx

explainer = ClassifierExplainer(model,
                                X_test,
                                y_test,
                                labels=["No Diabetes", "Diabetes"])
explainer.to_html("report.html")
```
To render your explainability report to html:
```python
from clinicalxai.generate_report import render_report

render_report(explainer,
              output_path="./diabetes_XAI_report.html",
              title="Diabetes Prediction Model XAI Report")
```
