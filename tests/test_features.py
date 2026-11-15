import numpy as np
import pandas as pd

from credit_risk.features import CATEGORICAL_FEATURES, MODEL_FEATURES, make_preprocessor


def test_preprocessing_learns_only_training_statistics_and_handles_unknown_categories():
    train = pd.DataFrame(1.0, index=range(4), columns=MODEL_FEATURES)
    train["LIMIT_BAL"] = [10.0, 20.0, np.nan, 30.0]
    holdout = train.iloc[[0]].copy()
    holdout["LIMIT_BAL"] = 1_000_000.0
    holdout[CATEGORICAL_FEATURES] = 99
    holdout["ID"] = 999
    holdout["DEFAULT"] = 1

    preprocessor = make_preprocessor(scale_numeric=True)
    preprocessor.fit(train)
    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[0] == 20.0
    before = imputer.statistics_.copy()
    vocabulary = preprocessor.named_transformers_["categorical"].named_steps["one_hot"]
    transformed = preprocessor.transform(holdout)

    np.testing.assert_equal(imputer.statistics_, before)
    assert np.isfinite(transformed).all()
    assert all(99 not in categories for categories in vocabulary.categories_)
    assert "ID" not in preprocessor.get_feature_names_out()
    assert "DEFAULT" not in preprocessor.get_feature_names_out()
