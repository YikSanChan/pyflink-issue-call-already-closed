from pyflink.table.udf import udf
from pyflink.table import DataTypes


@udf(result_type=DataTypes.INT(), func_type="pandas")
def label_encode(x, name):
    import pickle
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    with open('lbe.pkl', 'rb') as f:
        lbes = pickle.load(f)
        return pd.Series(lbes[name].transform(x))