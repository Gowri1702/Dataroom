import pandas as pd


def load_csv(uploaded_csv):

    df = pd.read_csv(uploaded_csv)
    return df


def profile_csv(df):

    profile = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "column_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist()
    }

    return profile