import pandas as pd
import numpy as np
import warnings
import joblib
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')


class HotelBookingModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.label_encoders = {}
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        return self.df

    def preprocess_data(self):
        df = self.df.copy()
        
        if 'Booking_ID' in df.columns:
            df = df.drop(columns='Booking_ID')

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        for col in df.select_dtypes(exclude=['object']).columns:
            df[col] = df[col].fillna(df[col].median())

        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        X = df.drop(columns='booking_status')
        y = df['booking_status']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

    def train_model(self, model_type='random_forest'):
        if model_type == 'xgboost':
            self.model = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                max_depth=5,
                n_estimators=50
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=10,
                random_state=42,
                max_depth=8
            )
        else:
            raise ValueError("Unsupported model type. Use 'xgboost' or 'random_forest'.")

        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", acc)
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        return acc

    def save_model(self, filename='best_booking_model.pkl'):
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")


if __name__ == "__main__":
    model_choice = 'random_forest'
    if len(sys.argv) > 1:
        model_choice = sys.argv[1]

    model = HotelBookingModel("Dataset_B_hotel.csv")
    model.load_data()
    model.preprocess_data()
    model.train_model(model_choice)
    model.evaluate_model()
    model.save_model("best_booking_model.pkl")

