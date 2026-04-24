from clinicalxai.ethical_eval import flag_top_features

def test_string_matching_is_case_insensitive_and_ignores_underscores_and_hyphens():
    top_features = ["ZipCode", "income", "Education_Level", "native-language", "EmploymentStatus", "occupation", "marital_status", "housing_status", "disability"]
    flagged = flag_top_features(top_features)

    expected_categories = {"socioeconomic_factors"}
    assert all(flag["category"] in expected_categories for flag in flagged)

def test_unknown_features_are_not_flagged():
    top_features = ["unknown_feature1", "unknown_feature2"]
    flagged = flag_top_features(top_features)

    assert len(flagged) == 0

def test_multiple_categories_can_be_flagged():
    top_features = ["ZipCode", "num visits", "age"]
    flagged = flag_top_features(top_features)

    expected_categories = {"socioeconomic_factors", "proxy_variables", "sociodemographic_factors"}
    flagged_categories = {flag["category"] for flag in flagged}

    assert flagged_categories == expected_categories

def test_flag_multiple_top_features_same_category():
    top_features = ["ZipCode", "income", "Education_Level"]
    flagged = flag_top_features(top_features)

    expected_categories = {"socioeconomic_factors"}
    flagged_categories = {flag["category"] for flag in flagged}
    assert flagged_categories == expected_categories
    assert len(flagged) == 3