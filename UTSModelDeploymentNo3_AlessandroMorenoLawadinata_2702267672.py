import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

class HotelBookingPredictor:
    def __init__(self, model_path, sample_data_path):
        self.model = self.load_model(model_path)
        self.label_encoders = {}
        self.columns = None
        self.load_label_encoders(sample_data_path)

    def load_model(self, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        return model

    def load_label_encoders(self, sample_data_path):
        """Gunakan data training untuk menyesuaikan label encoding"""
        df = pd.read_csv(sample_data_path)

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        for col in df.select_dtypes(exclude=['object']).columns:
            df[col] = df[col].fillna(df[col].median())
        
        self.columns = df.drop(columns='booking_status').columns.tolist()

        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

    def preprocess_input(self, input_dict):
        """Terima input sebagai dictionary dan ubah jadi dataframe"""
        input_df = pd.DataFrame([input_dict])
        
        
        for col in input_df.columns:
            if input_df[col].isnull().any():
                if input_df[col].dtype == 'object':
                    input_df[col].fillna(input_df[col].mode()[0], inplace=True)
                else:
                    input_df[col].fillna(input_df[col].median(), inplace=True)

        
        for col, le in self.label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        
        input_df = input_df[self.columns]
        return input_df

    def predict(self, input_dict):
        processed_input = self.preprocess_input(input_dict)
        prediction = self.model.predict(processed_input)
        return prediction[0]



if __name__ == '__main__':
    
    predictor = HotelBookingPredictor('best_booking_model.pkl', 'Dataset_B_hotel.csv')

    
    new_data = {
        'no_of_adults': 2,
        'no_of_children': 0,
        'no_of_weekend_nights': 1,
        'no_of_week_nights': 2,
        'type_of_meal_plan': 'Meal Plan 1',
        'required_car_parking_space': 0,
        'room_type_reserved': 'Room_Type 1',
        'lead_time': 34,
        'arrival_year': 2023,
        'arrival_month': 4,
        'arrival_date': 14,
        'market_segment_type': 'Online',
        'repeated_guest': 0,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 100,
        'no_of_special_requests': 1
    }

    result = predictor.predict(new_data)
    print("Booking Status Prediction:", "Confirmed" if result == 1 else "Canceled")
