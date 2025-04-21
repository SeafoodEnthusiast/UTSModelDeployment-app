import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os

class HotelBookingPredictor:
    def __init__(self, model_path, sample_data_path):
        self.model = self.load_model(model_path)
        self.label_encoders = {}
        self.columns = None
        self.load_label_encoders(sample_data_path)

    def load_model(self, path):
        return joblib.load(path)

    def load_label_encoders(self, sample_data_path):
        df = pd.read_csv(sample_data_path)

        
        if 'Booking_ID' in df.columns:
            df = df.drop(columns='Booking_ID')

        
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

MODEL_PATH = "best_booking_model.pkl"
DATA_PATH = "Dataset_B_hotel.csv"


if not os.path.exists(MODEL_PATH):
    st.error(f"File model '{MODEL_PATH}' tidak ditemukan.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset '{DATA_PATH}' tidak ditemukan.")
    st.stop()

predictor = HotelBookingPredictor(MODEL_PATH, DATA_PATH)

st.subheader("Isi data pemesanan hotel:")

input_data = {
    'no_of_adults': st.number_input("Jumlah Dewasa", 1, 10, 2),
    'no_of_children': st.number_input("Jumlah Anak", 0, 10, 0),
    'no_of_weekend_nights': st.number_input("Weekend Nights", 0, 10, 1),
    'no_of_week_nights': st.number_input("Week Nights", 0, 10, 2),
    'type_of_meal_plan': st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]),
    'required_car_parking_space': st.selectbox("Perlu Parkir?", [0, 1]),
    'room_type_reserved': st.selectbox("Tipe Kamar", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]),
    'lead_time': st.slider("Lead Time (hari)", 0, 500, 34),
    'arrival_year': st.selectbox("Tahun Kedatangan", [2023, 2024]),
    'arrival_month': st.slider("Bulan Kedatangan", 1, 12, 4),
    'arrival_date': st.slider("Tanggal Kedatangan", 1, 31, 14),
    'market_segment_type': st.selectbox("Segment Pasar", ["Online", "Offline", "Corporate", "Aviation", "Complementary"]),
    'repeated_guest': st.selectbox("Tamu Berulang?", [0, 1]),
    'no_of_previous_cancellations': st.number_input("Jumlah Pembatalan Sebelumnya", 0, 20, 0),
    'no_of_previous_bookings_not_canceled': st.number_input("Jumlah Booking Tidak Dibatalkan", 0, 20, 0),
    'avg_price_per_room': st.slider("Harga Rata-rata per Kamar (€)", 0, 1000, 100),
    'no_of_special_requests': st.number_input("Jumlah Permintaan Khusus", 0, 5, 1)
}

if st.button("Prediksi"):
    try:
        prediction = predictor.predict(input_data)
        status = "Booking Dikonfirmasi" if prediction == 1 else "Booking Dibatalkan"
        st.success(f"Hasil Prediksi: **{status}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")


st.markdown("---")
st.subheader("Contoh Test Case")

if st.button("Test Case 1 (Dikonfirmasi)"):
    st.session_state.update({
        'no_of_adults': 2,
        'no_of_children': 0,
        'no_of_weekend_nights': 2,
        'no_of_week_nights': 3,
        'type_of_meal_plan': "Meal Plan 1",
        'required_car_parking_space': 1,
        'room_type_reserved': "Room_Type 1",
        'lead_time': 14,
        'arrival_year': 2024,
        'arrival_month': 5,
        'arrival_date': 15,
        'market_segment_type': "Online",
        'repeated_guest': 1,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 2,
        'avg_price_per_room': 120,
        'no_of_special_requests': 2
    })
    st.experimental_rerun()

if st.button("Test Case 2 (Dibatalkan)"):
    st.session_state.update({
        'no_of_adults': 1,
        'no_of_children': 2,
        'no_of_weekend_nights': 0,
        'no_of_week_nights': 1,
        'type_of_meal_plan': "Not Selected",
        'required_car_parking_space': 0,
        'room_type_reserved': "Room_Type 6",
        'lead_time': 300,
        'arrival_year': 2023,
        'arrival_month': 11,
        'arrival_date': 3,
        'market_segment_type': "Offline",
        'repeated_guest': 0,
        'no_of_previous_cancellations': 2,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 80,
        'no_of_special_requests': 0
    })
    st.experimental_rerun()



