from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    Splits dataset into train and test sets.
    """
    return train_test_split(df["url"], df["label"], test_size=test_size, random_state=42)

