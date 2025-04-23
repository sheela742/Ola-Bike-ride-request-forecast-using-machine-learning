import tkinter as tk
from tkinter import messagebox, Toplevel
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
file_path = r"C:\Users\veda1\OneDrive\Desktop\ML-PRJ\ride_data.csv"
df = pd.read_csv(file_path, skipinitialspace=True)
df.columns = df.columns.str.strip()
df["datetime"] = pd.to_datetime(df["datetime"])
df["hour"] = df["datetime"].dt.hour

# Encode traffic column
label_encoder = LabelEncoder()
df["traffic"] = label_encoder.fit_transform(df["traffic"])

# Define features and target variable
X = df[["hour", "temp", "humidity", "traffic"]]
y = df["ride_requests"]  # No MinMaxScaler to keep real values

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_split=4)
model.fit(X_train, y_train)

# Function to predict ride requests
def predict_rides():
    try:
        hour = int(hour_entry.get())
        temp = float(temp_entry.get())
        humidity = float(humidity_entry.get())
        traffic = traffic_var.get().lower()
        traffic_encoded = label_encoder.transform([traffic])[0]

        user_input = pd.DataFrame([[hour, temp, humidity, traffic_encoded]], columns=["hour", "temp", "humidity", "traffic"])
        user_input_scaled = scaler.transform(user_input)
        predicted_rides = model.predict(user_input_scaled)[0]  # Direct prediction without scaling

        # Decision Logic based on actual predicted values
        if predicted_rides >= 200:
            ride_message = "✅ High Demand for Rides! 🚴 Enjoy your trip!"
        elif 100 <= predicted_rides < 200:
            ride_message = "🟡 Moderate Demand, Rides Available!"
        elif 50 <= predicted_rides < 100:
            ride_message = "🟠 Low Demand, You May Have to Wait!"
        
        # Show result in a new window
        result_window = Toplevel(root)
        result_window.title("Prediction Result")
        result_window.geometry("650x450")
        result_window.configure(bg="#34495E")

        tk.Label(result_window, text="Ride Demand Prediction", font=("Arial", 22, "bold"), fg="white", bg="#34495E").pack(pady=20)
        tk.Label(result_window, text=f"Predicted Ride Requests: {predicted_rides:.2f}", font=("Arial", 18), fg="yellow", bg="#34495E").pack(pady=10)
        tk.Label(result_window, text=ride_message, font=("Arial", 16), fg="lightblue", bg="#34495E", wraplength=450, justify="center").pack(pady=10)

        tk.Button(result_window, text="Close", command=result_window.destroy, font=("Arial", 16), bg="#C0392B", fg="white", padx=20, pady=10).pack(pady=20)

    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}")

# Tkinter GUI
root = tk.Tk()
root.title("OLA Bike Ride Request Forecast")
root.configure(bg="#2E4053")
root.attributes('-fullscreen', True)

# Title
title_label = tk.Label(root, text="OLA Bike Ride Request Forecast", font=("Arial", 30, "bold"), fg="white", bg="#2E4053")
title_label.pack(pady=20)

# Main frame
frame = tk.Frame(root, bg="#2E4053", padx=30, pady=30)
frame.pack(expand=True)

# Labels & Entry Fields
def create_label_entry(label_text):
    label = tk.Label(frame, text=label_text, font=("Arial", 18), fg="white", bg="#2E4053")
    label.pack(pady=10)
    entry = tk.Entry(frame, font=("Arial", 18), width=15)
    entry.pack(pady=5)
    return entry

hour_entry = create_label_entry("Hour (0-23):")
temp_entry = create_label_entry("Temperature (°C):")
humidity_entry = create_label_entry("Humidity (%):")

# Traffic Dropdown
tk.Label(frame, text="Traffic Level:", font=("Arial", 18), fg="white", bg="#2E4053").pack(pady=10)
traffic_var = tk.StringVar()
traffic_var.set("low")

traffic_dropdown = tk.OptionMenu(frame, traffic_var, "low", "medium", "high")
traffic_dropdown.config(font=("Arial", 16), width=12, bg="white")
traffic_dropdown.pack(pady=5)

# Predict Button
predict_button = tk.Button(frame, text="Predict Ride Demand", command=predict_rides, font=("Arial", 20, "bold"), bg="#28B463", fg="white", padx=20, pady=10)
predict_button.pack(pady=20)

# Exit Button
exit_button = tk.Button(root, text="Exit", command=root.destroy, font=("Arial", 16), bg="#C0392B", fg="white", padx=20, pady=10)
exit_button.pack(pady=20)

root.mainloop()
