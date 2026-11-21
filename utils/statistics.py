import pandas as pd


def describe(frame):
    columns = frame.select_dtypes("number")
    result = columns.describe().T
    result["median"] = columns.median()
    result["mode"] = pd.Series({
        name: modes.iloc[0] if len(modes := values.mode()) else float("nan")
        for name, values in columns.items()
    })
    result["range"] = result["max"] - result["min"]
    result["variance"] = columns.var()
    result["IQR"] = result["75%"] - result["25%"]
    result["skewness"] = columns.skew()
    result["kurtosis"] = columns.kurt()
    result["missing %"] = columns.isna().mean() * 100
    return result
