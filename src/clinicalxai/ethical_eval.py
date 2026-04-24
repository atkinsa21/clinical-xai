"""
Ethical evaluation component of Clinical XAI final report.
"""
from typing import Iterable
import re


_FLAGS: dict[str, dict] = {
    "socioeconomic_factors": {
        "severity": "warning",
        "description": "Top feature is a measure of socioeconomic status. This may reflect underlying health disparities and could lead to biased predictions.",
        "mitigation": "Consider removing socioeconomic features or using techniques to mitigate bias, such as reweighting or adversarial training.",
        "keywords": ["zipcode", "income", "educationlevel", "nativelanguage", "employmentstatus", "occupation", "maritalstatus", "housingstatus", "disability"],
    },
    "proxy_variables": {
        "severity": "warning",
        "description": "Top feature is a proxy for a protected attribute (e.g., ICD codes, number of visits, healthcare costs).",
        "mitigation": "Verify the model's reliance on the feature and consider removing it if it's a proxy. Substitute with direct clinical measures where possible.",
        "keywords": ["icdcode", "numvisits", "healthcarecosts", "insurance", "healthcarespending"],
    },
    "sociodemographic_factors": {
        "severity": "warning",
        "description": "Top feature is a sociodemographic factor (e.g., age, sex, race, ethnicity).",
        "mitigation": "Ensure the model is not unfairly biased against certain groups. Consider stratified performance evaluation and fairness metrics.",
        "keywords": ["age", "sex", "gender", "race", "ethnicity", "countryoforigin"],
    }
}

def flag_top_features(top_features: Iterable[str]) -> list[dict]:
    """
    Returns one entry per string-matched feature (lowercase, stripped underscores/hyphens): {feature, category, severity, rationale, mitigation}
    First matching category wins (one category per feature)
    """
    flagged_features = []
    for feature in top_features:
        feature_stripped = re.sub(r'[^a-zA-Z]', '', feature.lower())
        for category, details in _FLAGS.items():
            if feature_stripped in details["keywords"]:
                flagged_features.append({
                    "feature": feature,
                    "category": category,
                    "severity": details["severity"],
                    "rationale": details["description"],
                    "mitigation": details["mitigation"]
                })
    return flagged_features