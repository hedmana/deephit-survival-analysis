import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def preprocess_metabrics(df: pd.DataFrame) -> pd.DataFrame:

    # Drop all rows with missing values
    df = df.dropna().reset_index(drop=True)

    # Rename for consistency
    df.rename(
        columns={"overall_survival_months": "time", "overall_survival": "event"},
        inplace=True,
    )

    # Encode categorical features (if any)
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Normalize continuous features
    cont_cols = [
        "age_at_diagnosis",
        "lymph_nodes_examined_positive",
        "nottingham_prognostic_index",
    ]
    df[cont_cols] = (df[cont_cols] - df[cont_cols].mean()) / df[cont_cols].std()

    return df


def discretize_time(t, num_bins=100):
    max_time = np.max(t)
    t = np.clip(t, 0, max_time)
    t_discrete = np.floor((t / max_time) * (num_bins - 1)).astype(int)
    return t_discrete, num_bins


def make_mask(t, e, num_Event, num_Category):
    n = len(t)
    mask1 = np.zeros((n, num_Event, num_Category))
    mask2 = np.zeros((n, num_Category))

    for i in range(n):
        tt = int(t[i])
        ee = int(e[i])
        if ee > 0:  # uncensored
            mask1[i, ee - 1, tt] = 1
        mask2[i, tt + 1 :] = 1  # P(T > t)

    return mask1, mask2


def prepare_deephit_data(df, num_bins=100, test_size=0.2, random_state=42):
    x = df.drop(columns=["patient_id", "time", "event"]).values
    t = df["time"].values
    e = df["event"].values

    # Discretize time
    t_disc, num_Category = discretize_time(t, num_bins=num_bins)

    # Train/test split
    x_train, x_test, t_train, t_test, e_train, e_test = train_test_split(
        x, t_disc, e, test_size=test_size, random_state=random_state
    )

    # Create masks
    m1_train, m2_train = make_mask(
        t_train, e_train, num_Event=1, num_Category=num_Category
    )
    m1_test, m2_test = make_mask(t_test, e_test, num_Event=1, num_Category=num_Category)

    data = {
        "train": {
            "x": x_train,
            "t": t_train.reshape(-1, 1),
            "e": e_train.reshape(-1, 1),
            "m1": m1_train,
            "m2": m2_train,
        },
        "test": {
            "x": x_test,
            "t": t_test.reshape(-1, 1),
            "e": e_test.reshape(-1, 1),
            "m1": m1_test,
            "m2": m2_test,
        },
        "meta": {
            "x_dim": x.shape[1],
            "num_Event": 1,
            "num_Category": num_Category,
        },
    }

    return data
