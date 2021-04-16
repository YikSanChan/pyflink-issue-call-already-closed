import numpy as np
import pandas as pd

from ml_udf import label_encode


def test_label_encode():
    f = label_encode._func
    encoded = f(pd.Series([0, 1, 2]), "source")
    assert isinstance(encoded, pd.Series)
    assert len(encoded) == 3
