import streamlit as st
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="EV Battery Monitoring System",
    page_icon="⚡",
    layout="wide"
)

# ====================================
# CUSTOM CSS
# ====================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b
    );
}

h1,h2,h3,h4,h5,h6,p,label {
    color:white !important;
}

section[data-testid="stSidebar"] {
    background-color:#111827;
}

</style>
""", unsafe_allow_html=True)

# ====================================
# SIDEBAR
# ====================================

st.sidebar.title("⚡ EV Analytics")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Admin Panel",
        "SOH Prediction",
        "Range Calculation",
        "Nearest Charging Station"
    ]
)

# ====================================
# HOME PAGE
# ====================================

if menu == "Home":

    st.title(
        "⚡ EV Battery Health Monitoring System"
    )

    st.markdown("""
    ### Machine Learning Based EV Analytics Platform
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "ML Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "Prediction",
            "SOH"
        )

    with col3:
        st.metric(
            "Navigation",
            "EV Stations"
        )

    st.markdown("---")

    st.subheader("Project Features")

    st.write("✅ Battery State Of Health Prediction")

    st.write("✅ Battery Health Grade")

    st.write("✅ Remaining Range Estimation")

    st.write("✅ Nearest Charging Station Finder")

    st.write("✅ Streamlit Cloud Deployment")

# ====================================
# ADMIN PANEL
# ====================================

elif menu == "Admin Panel":

    st.title("📊 Admin Panel")

    uploaded_file = st.file_uploader(
        "Upload Battery Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

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

            y = df["SOH"]

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
                "✅ Model Trained Successfully"
            )

# ====================================
# SOH PREDICTION
# ====================================

elif menu == "SOH Prediction":

    st.title("🔋 Battery SOH Prediction")

    cycle = st.number_input(
        "Cycle Count",
        min_value=0
    )

    resistance = st.number_input(
        "Internal Resistance",
        min_value=0.0
    )

    capacity = st.number_input(
        "Capacity",
        min_value=0.0
    )

    temperature = st.number_input(
        "Temperature",
        min_value=-50.0
    )

    voltage = st.number_input(
        "Voltage",
        min_value=0.0
    )

    if st.button("Predict SOH"):

        try:

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

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted SOH",
                    f"{soh:.2f}%"
                )

            with col2:
                st.metric(
                    "Battery Grade",
                    grade
                )

        except:
            st.error(
                "Train model first in Admin Panel"
            )

# ====================================
# RANGE CALCULATION
# ====================================

elif menu == "Range Calculation":

    st.title(
        "🚗 Estimated Range Calculation"
    )

    soh = st.number_input(
        "SOH (%)",
        value=90.0
    )

    battery_percent = st.slider(
        "Battery Percentage",
        0,
        100,
        50
    )

    battery_capacity = st.number_input(
        "Battery Capacity (kWh)",
        value=50.0
    )

    efficiency = st.number_input(
        "Efficiency (km/kWh)",
        value=6.0
    )

    if st.button(
        "Calculate Range"
    ):

        usable_energy = (
            battery_capacity
            *
            (battery_percent/100)
            *
            (soh/100)
        )

        range_km = (
            usable_energy
            *
            efficiency
        )

        st.metric(
            "Estimated Range",
            f"{range_km:.2f} km"
        )

# ====================================
# CHARGING STATION
# ====================================

elif menu == "Nearest Charging Station":

    st.title(
        "⚡ Nearest Charging Station"
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
            lambda row:
            geodesic(
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
            "Top 5 Nearest Charging Stations"
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

# ====================================
# FOOTER
# ====================================

st.markdown("---")

st.markdown(
    """
    <center>
    ⚡ EV Battery Health Monitoring System<br>
    Developed by Mukesh M<br>
    PG Project
    </center>
    """,
    unsafe_allow_html=True
)