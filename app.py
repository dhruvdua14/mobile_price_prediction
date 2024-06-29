# Create a Streamlit App (mobile_class_app.py)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load the pickled model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Define the Streamlit app
def main():
    st.title("Mobile Price Range Classifier")

    # Define the image path
    image_path = os.path.join("data", "images", "phone1_image.jpg")

    # Add picture of phone display
    st.image(image_path, caption="Image source: Google", use_column_width=True)

    # User input for mobile features
    st.sidebar.title("Feature Selection:")

    feature_descriptions = {
        "battery_power": "Total energy a battery can store in mAh (min_value = 501, max_value = 1998).",
        "blue": "Bluetooth enabled.",
        "clock_speed": "Speed at which the microprocessor executes instructions (min_value = 0.5, max_value = 3.0).",
        "dual_sim": "Dual SIM support ",
        "fc": "Front Camera mega pixels (min_value = 0.0, max_value = 19).",
        "four_g": "4G network support.",
        "int_memory": "Internal Memory in gigabytes (min_value = 2, max_value = 64).",
        "m_dep": "Mobile Depth in cm (min_value = 0.1, max_value = 1.0).",
        "mobile_wt": "Weight of mobile phone (min_value = 80, max_value = 200).",
        "n_cores": "Number of cores of the processor (min_value = 1, max_value = 8).",
        "pc": "Primary Camera mega pixels (min_value = 0, max_value = 20).",
        "px_height": "Pixel Resolution Height (min_value = 0, max_value = 1960).",
        "px_width": "Pixel Resolution Width (min_value = 500, max_value = 1998).",
        "ram": "Random Access Memory in megabytes (min_value = 256, max_value = 3998).",
        "sc_h": "Screen Height of mobile in cm (min_value = 5, max_value = 19).",
        "sc_w": "Screen Width of mobile in cm (min_value = 0, max_value = 18).",
        "talk_time": "Longest time that a single battery charge will last when talking (min_value = 2, max_value = 20).",
        "three_g": "3G network support.",
        "touch_screen": "Touch screen support.",
        "wifi": "Wifi connectivity."
    }

    categorical_mapping = {1: "Yes", 0: "No"}

    # Add Streamlit components for user input with feature descriptions
    battery_power = st.sidebar.number_input("Battery Power (mAh)", min_value=501, max_value=1998,
                                            help=feature_descriptions.get("battery_power", ""))

    blue = st.sidebar.radio("Bluetooth Availability", list(categorical_mapping.values()),
                            help=feature_descriptions.get("blue", ""))

    clock_speed = st.sidebar.number_input("Clock Speed (GHz)", min_value=0.5, max_value=3.0,
                                          help=feature_descriptions.get("clock_speed", ""))

    dual_sim = st.sidebar.radio("Dual SIM Support", list(categorical_mapping.values()),
                                help=feature_descriptions.get("dual_sim", ""))

    fc = st.sidebar.number_input("Front Camera (MP)", min_value=0, max_value=19,
                                 help=feature_descriptions.get("fc", ""))

    four_g = st.sidebar.radio("4G enabled?", list(categorical_mapping.values()),
                              help=feature_descriptions.get("four_g", ""))

    int_memory = st.sidebar.number_input("Internal Memory (GB)", min_value=2, max_value=64,
                                         help=feature_descriptions.get("int_memory", ""))

    m_dep = st.sidebar.number_input("Mobile Depth (cm)", min_value=0.1, max_value=1.0,
                                    help=feature_descriptions.get("m_dep", ""))

    mobile_wt = st.sidebar.number_input("Mobile Weight", min_value=80, max_value=200,
                                        help=feature_descriptions.get("mobile_wt", ""))

    n_cores = st.sidebar.number_input("Number of Cores", min_value=1, max_value=8,
                                      help=feature_descriptions.get("n_cores", ""))

    pc = st.sidebar.number_input("Primary Camera (MP)", min_value=0, max_value=20,
                                 help=feature_descriptions.get("pc", ""))

    px_height = st.sidebar.number_input("Pixel Resolution Height", min_value=0, max_value=1960,
                                        help=feature_descriptions.get("px_height", ""))

    px_width = st.sidebar.number_input("Pixel Resolution Width", min_value=500, max_value=1998,
                                       help=feature_descriptions.get("px_width", ""))

    ram = st.sidebar.number_input("Random Access Memory (MB)", min_value=256, max_value=3998,
                                  help=feature_descriptions.get("ram", ""))

    sc_h = st.sidebar.number_input("Screen Height (cm)", min_value=5, max_value=19,
                                   help=feature_descriptions.get("sc_h", ""))

    sc_w = st.sidebar.number_input("Screen Width (cm)", min_value=0, max_value=18,
                                   help=feature_descriptions.get("sc_w", ""))

    talk_time = st.sidebar.number_input("Talk Time", min_value=2, max_value=20,
                                        help=feature_descriptions.get("talk_time", ""))

    three_g = st.sidebar.radio("3G enabled?", list(categorical_mapping.values()),
                               help=feature_descriptions.get("three_g", ""))

    touch_screen = st.sidebar.radio("Touchscreen availability?", list(categorical_mapping.values()),
                                    help=feature_descriptions.get("touch_screen", ""))

    wifi = st.sidebar.radio("Wi-Fi availability?", list(categorical_mapping.values()),
                            help=feature_descriptions.get("wifi", ""))

    # Make predictions based on user input
    if st.sidebar.button('Predict'):
        # Convert categorical features to numerical values
        blue_num = 1 if blue == "Yes" else 0
        dual_sim_num = 1 if dual_sim == "Yes" else 0
        four_g_num = 1 if four_g == "Yes" else 0
        three_g_num = 1 if three_g == "Yes" else 0
        touch_screen_num = 1 if touch_screen == "Yes" else 0
        wifi_num = 1 if wifi == "Yes" else 0

        input_df = pd.DataFrame({
            "battery_power": [battery_power],
            "blue": [blue_num],
            "clock_speed": [clock_speed],
            "dual_sim": [dual_sim_num],
            "fc": [fc],
            "four_g": [four_g_num],
            "int_memory": [int_memory],
            "m_dep": [m_dep],
            "mobile_wt": [mobile_wt],
            "n_cores": [n_cores],
            "pc": [pc],
            "px_height": [px_height],
            "px_width": [px_width],
            "ram": [ram],
            "sc_h": [sc_h],
            "sc_w": [sc_w],
            "talk_time": [talk_time],
            "three_g": [three_g_num],
            "touch_screen": [touch_screen_num],
            "wifi": [wifi_num]
        })

        # Make predictions
        prediction = model.predict(input_df)

        # Map predicted values to their corresponding labels
        price_range_labels = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
        predicted_label = price_range_labels.get(prediction[0], "Unknown")

        # Display the predicted price range
        st.success(f'Predicted Price Range: {predicted_label}')
        st.balloons()


# 3. Deploy the Streamlit App
if __name__ == "__main__":
    main()