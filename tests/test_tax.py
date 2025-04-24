
import pandas as pd
from services.tax import TaxService

def test_compute_for_row_basic():
    service = TaxService()

    row = pd.Series({
        "syear": 2021,
        "parental_income_post_insurance_allowance": 40_000,
        "bula": 1,             # not church-tax Länder
        "plh0258_h": 3,        # no church membership
        "hgtyp1hh": 1          # single filing
    })

    itax, church, soli = service.compute_for_row(row)

    assert itax is not None and itax > 0
    assert church == 0          # no church tax
    assert soli >= 0
