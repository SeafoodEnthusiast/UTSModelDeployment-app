# UTSModelDeployment-app

# Hotel Booking Cancellation Prediction

An end-to-end machine learning project that predicts whether a hotel booking will be cancelled, from raw data through to a deployed Streamlit web application.

**[Live Demo](https://utsmodeldeployment-app-gbs2bah3cg3xpubngca83i.streamlit.app)** · **[Video Walkthrough](https://youtu.be/WgpLeYn3mrI)**

---

## Problem

Cancellations are a persistent revenue problem for hotels. A reservation that falls through too late to resell means an empty room, while overly aggressive overbooking risks turning away guests who actually show up.

If a hotel can estimate cancellation risk at the moment a booking is made, it can act on that signal — adjusting overbooking limits, targeting retention offers at high-risk reservations, or asking for deposits selectively.

This project trains a classifier to predict a reservation's `booking_status` using only information available at booking time, and deploys it as a web app where a user enters booking details and gets an immediate prediction.

## Dataset

`Dataset_B_hotel.csv` — 36,275 hotel reservations with 17 predictive features:

| Category | Features |
|---|---|
| Guests | `no_of_adults`, `no_of_children`, `repeated_guest` |
| Stay | `no_of_weekend_nights`, `no_of_week_nights`, `room_type_reserved`, `type_of_meal_plan` |
| Timing | `lead_time`, `arrival_year`, `arrival_month`, `arrival_date` |
| History | `no_of_previous_cancellations`, `no_of_previous_bookings_not_canceled` |
| Commercial | `avg_price_per_room`, `market_segment_type`, `required_car_parking_space`, `no_of_special_requests` |

**Target:** `booking_status` — 1 for a confirmed booking, 0 for a cancelled one. The classes are imbalanced roughly 2:1 in favour of confirmed bookings.

Three columns had missing values:

| Column | Missing |
|---|---|
| `avg_price_per_room` | 1,632 |
| `required_car_parking_space` | 1,270 |
| `type_of_meal_plan` | 907 |

## Approach

**Data understanding**

Before changing anything, I checked data types, descriptive statistics, category counts, missing values, and outliers using the IQR rule. The outlier check turned out to matter: `no_of_adults` alone flagged over 10,000 rows, and `avg_price_per_room` around 1,600.

**Preprocessing**

- Dropped `Booking_ID`, an identifier with no predictive value.
- Imputed `type_of_meal_plan` and `required_car_parking_space` with the mode, and `avg_price_per_room` with the median.
- Applied IQR clipping (winsorizing) to the ten columns flagged as having outliers. This brought the outlier count to zero without dropping a single row, which mattered because the flagged values were legitimate bookings rather than data errors.
- Encoded categorical features with `LabelEncoder`, keeping each fitted encoder so that user input at inference time is transformed exactly the way training data was.
- Split 80/20 into train and test sets with a fixed random seed for reproducibility.

**Model comparison**

Both candidates were trained on the same split and evaluated with accuracy and a full classification report:

| Model | Configuration |
|---|---|
| Random Forest | `n_estimators=10` |
| XGBoost | `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric='logloss'` |

**Refactoring**

The training workflow was restructured from notebook cells into a `HotelBookingModel` class exposing `load_data()`, `preprocess_data()`, `train_model()`, `evaluate_model()`, and `save_model()`. The model type is selectable from the command line, so retraining with a different algorithm is a one-line change rather than an edit to the pipeline.

**Deployment**

A separate `HotelBookingPredictor` class loads the serialised model and the fitted encoders, then serves predictions through a Streamlit interface with form inputs for all 17 features.

## Results

| Model | Accuracy |
|---|---|
| **Random Forest** (selected) | **89.3%** |
| XGBoost | 88.3% |

The one-point gap in accuracy understates the difference between the two. Recall on the cancelled class — the class a hotel actually cares about — separates them more clearly:

| Class | Model | Precision | Recall | Support |
|---|---|---|---|---|
| Cancelled (0) | Random Forest | 0.84 | **0.83** | 2,416 |
| Cancelled (0) | XGBoost | 0.85 | **0.78** | 2,416 |
| Confirmed (1) | Random Forest | 0.92 | 0.92 | 4,839 |
| Confirmed (1) | XGBoost | 0.90 | 0.93 | 4,839 |

XGBoost is marginally more precise when it does predict a cancellation, but it predicts far fewer of them: it misses 22% of real cancellations against Random Forest's 17%. Since a missed cancellation is the expensive error here, Random Forest was selected and serialised to `best_booking_model.pkl`.

Both models are weaker on the cancelled class than the confirmed one, which follows from the 2:1 class imbalance — the majority class is simply easier to learn.

## Repository Structure

```
├── Dataset_B_hotel.csv                                                 # Raw dataset
├── UTSModelDeploymentNo1_AlessandroMorenoLawadinata_2702267672.ipynb   # EDA, preprocessing, model comparison
├── UTSModelDeploymentNo2_AlessandroMorenoLawadinata_2702267672.py      # Training pipeline (OOP)
├── UTSModelDeploymentNo3_AlessandroMorenoLawadinata_2702267672.py      # Inference class + Streamlit app
├── best_booking_model.pkl                                              # Serialised Random Forest model
├── requirements.txt                                                    # Dependencies
└── .devcontainer/                                                      # Dev container config
```

## Running Locally

```bash
git clone https://github.com/SeafoodEnthusiast/UTSModelDeployment-app.git
cd UTSModelDeployment-app
pip install -r requirements.txt
```

Launch the web app:

```bash
streamlit run UTSModelDeploymentNo3_AlessandroMorenoLawadinata_2702267672.py
```

Retrain the model from scratch:

```bash
python UTSModelDeploymentNo2_AlessandroMorenoLawadinata_2702267672.py random_forest
# or
python UTSModelDeploymentNo2_AlessandroMorenoLawadinata_2702267672.py xgboost
```

## Notes on Some Decisions

I imputed numeric columns with the median rather than the mean because features like `lead_time` and `avg_price_per_room` are skewed, and the mean would have been dragged by the long tail.

I kept the fitted `LabelEncoder` objects on the model class instead of re-encoding at prediction time. Without this, a category the user selects in the app could be mapped to a different number than it was during training, and the model would silently give wrong answers.

Revisiting this project a year later, the thing I would change first is the evaluation. I reported accuracy because it was what the task asked for, but for a cancellation model recall matters more — missing a booking that actually cancels costs the hotel more than a false alarm does. Reading the per-class numbers again also showed something the headline accuracy had hidden: XGBoost is meaningfully worse at catching cancellations despite being only one point behind overall.

Other things worth doing:

- **Address the class imbalance.** Class weighting or resampling would likely lift recall on cancellations, at some cost to precision — a trade worth making here.
- **Grow the forest.** Ten trees is enough to reach 89%, but more estimators would cost little and give more stable predictions.
- **Persist the encoders alongside the model,** so deployment doesn't depend on the raw CSV being present at startup.
- **Return a probability instead of a hard label,** so a user can judge borderline cases rather than trusting a binary answer.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `joblib` · `Streamlit`

## Context

Built as the midterm project for the Model Deployment course, Data Science program, Bina Nusantara University — April 2025.

---

**Alessandro Moreno Lawadinata**
[LinkedIn](https://linkedin.com/in/alessandro-moreno-lawadinata) · [GitHub](https://github.com/SeafoodEnthusiast)
