import pandas as pd
import numpy as np
from pipeline import steps as S

def test_apply_social_insurance():
    df = pd.DataFrame({"adjusted_parental_income": [10_000.0]})
    out = S.apply_social_insurance_allowance(df)   # default 22.3 %
    assert np.isclose(out.loc[0, "parental_income_post_insurance_allowance"],
                      10_000 * (1 - 0.223))
