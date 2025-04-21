import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import streamlit as st

class HotelBookingPredictor:
    def __init__(self, model_path, sample_data_path):
        self.model = self.load_model(model_path)
        self.label_encoders = {}
        self.columns = None
        self.load_label_encoders(sample_data_path)

    def load_model(self, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def load_label_encoders(self, sample_data_path):
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



st.title("Hotel Booking Status Predictor")

predictor = HotelBookingPredictor('best_booking_model.pkl', 'Dataset_B_hotel.csv')

st.write("Silakan isi informasi pemesanan di bawah:")

input_data = {
    'no_of_adults': st.number_input("Jumlah Dewasa", min_value=1, max_value=10, value=2),
    'no_of_children': st.number_input("Jumlah Anak", min_value=0, max_value=10, value=0),
    'no_of_weekend_nights': st.number_input("Weekend Nights", min_value=0, max_value=10, value=1),
    'no_of_week_nights': st.number_input("Week Nights", min_value=0, max_value=10, value=2),
    'type_of_meal_plan': st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]),
    'required_car_parking_space': st.selectbox("Perlu Parkir?", [0, 1]),
    'room_type_reserved': st.selectbox("Tipe Kamar", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]),
    'lead_time': st.slider("Lead Time", 0, 500, 34),
    'arrival_year': st.selectbox("Tahun Kedatangan", [2023, 2024]),
    'arrival_month': st.slider("Bulan Kedatangan", 1, 12, 4),
    'arrival_date': st.slider("Tanggal Kedatangan", 1, 31, 14),
    'market_segment_type': st.selectbox("Segment Pasar", ["Online", "Offline", "Corporate", "Aviation", "Complementary"]),
    'repeated_guest': st.selectbox("Tamu Berulang?", [0, 1]),
    'no_of_previous_cancellations': st.number_input("Jumlah Pembatalan Sebelumnya", min_value=0, value=0),
    'no_of_previous_bookings_not_canceled': st.number_input("Jumlah Booking Tidak Dibatalkan", min_value=0, value=0),
    'avg_price_per_room': st.slider("Harga Rata-rata per Kamar", 0, 1000, 100),
    'no_of_special_requests': st.number_input("Jumlah Permintaan Khusus", min_value=0, max_value=5, value=1)
}

if st.button("Prediksi"):
    prediction = predictor.predict(input_data)
    status = "Confirmed" if prediction == 1 else "Canceled"
    st.success(f"Prediksi Status Booking: **{status}**")

