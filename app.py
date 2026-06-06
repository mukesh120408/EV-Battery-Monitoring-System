import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic

st.set_page_config(
    page_title="EV Battery Monitoring System",
    layout="wide"
)

# ==========================
# SIDEBAR
# ==========================

menu = st.sidebar.radio(
    "Navigation",
    [
        "Admin Panel",
        "SOH Prediction",
        "Range Calculation",
        "Nearest Charging Station"
    ]
)

# ==========================
# ADMIN PANEL
# ==========================

if menu == "Admin Panel":

    st.title("Admin Panel")

    uploaded_file = st.file_uploader(
        "Upload Battery Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write(df.head())

        if st.button("Train Model"):

            df["Resistance_Per_Cycle"] = (
                df["InternalResistance"]
                /
                (df["Cycle"] + 1)
            )

            features = [
                'Cycle',
                'InternalResistance',
                'Capacity',
                'Resistance_Per_Cycle',
                'Temperature',
                'Voltage'
            ]

            X = df[features]
            y = df['SOH']

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

            model.fit(
                X_train,
                y_train
            )

            joblib.dump(
                model,
                "soh_model.pkl"
            )

            st.success(
                "Model Trained Successfully"
            )

# ==========================
# SOH PREDICTION
# ==========================

elif menu == "SOH Prediction":

    st.title("SOH Prediction")

    cycle = st.number_input(
        "Cycle Count"
    )

    resistance = st.number_input(
        "Internal Resistance"
    )

    capacity = st.number_input(
        "Capacity"
    )

    temperature = st.number_input(
        "Temperature"
    )

    voltage = st.number_input(
        "Voltage"
    )

    if st.button("Predict SOH"):

        model = joblib.load(
            "soh_model.pkl"
        )

        resistance_per_cycle = (
            resistance /
            (cycle + 1)
        )

        data = [[
            cycle,
            resistance,
            capacity,
            resistance_per_cycle,
            temperature,
            voltage
        ]]

        soh = model.predict(
            data
        )[0]

        if soh >= 90:
            grade = "A"

        elif soh >= 80:
            grade = "B"

        elif soh >= 70:
            grade = "C"

        else:
            grade = "D"

        st.success(
            f"Predicted SOH : {soh:.2f}%"
        )

        st.info(
            f"Battery Grade : {grade}"
        )

# ==========================
# RANGE CALCULATION
# ==========================

elif menu == "Range Calculation":

    st.title(
        "Estimated Range Calculation"
    )

    soh = st.number_input(
        "Predicted SOH (%)"
    )

    battery_percent = st.slider(
        "Battery Percentage",
        0,
        100,
        50
    )

    battery_capacity = st.number_input(
        "Battery Capacity (kWh)",
        value=50
    )

    efficiency = st.number_input(
        "Vehicle Efficiency (km/kWh)",
        value=6
    )

    if st.button("Calculate Range"):

        usable_energy = (
            battery_capacity *
            (battery_percent/100) *
            (soh/100)
        )

        range_km = (
            usable_energy *
            efficiency
        )

        st.success(
            f"Estimated Range : {range_km:.2f} km"
        )

# ==========================
# NEAREST CHARGING STATION
# ==========================

elif menu == "Nearest Charging Station":

    st.title(
        "Nearest Charging Station"
    )

    station_file = st.file_uploader(
        "Upload Charging Station Dataset",
        type=["csv"]
    )

    user_lat = st.number_input(
        "Current Latitude",
        format="%.6f"
    )

    user_lon = st.number_input(
        "Current Longitude",
        format="%.6f"
    )

    if (
        station_file is not None
        and
        st.button(
            "Find Nearest Station"
        )
    ):

        stations = pd.read_csv(
            station_file
        )

        current_location = (
            user_lat,
            user_lon
        )

        stations["Distance_km"] = stations.apply(
            lambda row: geodesic(
                current_location,
                (
                    row["Latitude"],
                    row["Longitude"]
                )
            ).km,
            axis=1
        )

        nearest = stations.sort_values(
            by="Distance_km"
        ).head(5)

        st.subheader(
            "Top 5 Nearest Stations"
        )

        st.dataframe(
            nearest[
                [
                    "Station Name",
                    "City",
                    "Distance_km"
                ]
            ]
        )