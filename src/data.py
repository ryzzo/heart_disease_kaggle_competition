import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df

def split_xy_and_encode(df: pd.DataFrame, target_col: str, positive_label: str, negative_label: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    
    y_raw = df[target_col].astype(str).str.strip()
    X = df.drop(columns=[target_col])

    allowed = {positive_label, negative_label}
    observed = set(y_raw.unique())
    if not observed.issubset(allowed):
        raise ValueError(
            f"Unexpected labels in target. Observed: {sorted(observed)}."
            f"Allowed: {sorted(allowed)}"
        )
    
    # Encode: presence -> 1, absence -> 0
    y = y_raw.map({negative_label: 0, positive_label: 1}).astype(int)
    return X, y

def make_train_test(X, y, test_size: float, random_state: int):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)