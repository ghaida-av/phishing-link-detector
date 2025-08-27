import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning:
    - Drop missing URLs
    - Remove duplicates
    """
    df = df.dropna(subset=["url"])
    df = df.drop_duplicates(subset=["url"])
    return df

