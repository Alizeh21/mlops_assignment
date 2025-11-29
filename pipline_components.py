import kfp
from kfp.components import create_component_from_func
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load versioned data through DVC"""
    df = pd.read_csv("data/raw_data.csv")
    return df


def preprocess_data():
    """Preprocess the Boston housing dataset"""
    df = pd.read_csv("data/raw_data.csv")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    joblib.dump((X_train, X_test, y_train, y_test), "processed_data.pkl")
    return "processed_data.pkl"


def train_model(processed_data_path: str):
    """Train a regression model and save it"""
    X_train, X_test, y_train, y_test = joblib.load(processed_data_path)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")
    return "model.pkl"
