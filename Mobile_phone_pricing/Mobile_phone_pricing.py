import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("pricing.csv") 
X = df.drop("price_range", axis=1)
print(X.columns.tolist())
y = df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

joblib.dump(model, "price_predictor_model.pkl")
joblib.dump(scaler, "scaler.pkl")

def get_user_input():
    print("\nEnter the specifications of the mobile phone:")
    user_data = {
        'battery_power': int(input("Battery Power (in mAh): ")),
        'blue': int(input("Bluetooth (1 = Yes, 0 = No): ")),
        'clock_speed': float(input("Clock Speed (in GHz): ")),
        'dual_sim': int(input("Dual SIM (1 = Yes, 0 = No): ")),
        'fc': int(input("Front Camera (in MP): ")),
        'four_g': int(input("4G Support (1 = Yes, 0 = No): ")),
        'int_memory': int(input("Internal Memory (in GB): ")),
        'm_dep': float(input("Mobile Depth (in cm): ")),
        'mobile_wt': int(input("Weight (in grams): ")),
        'n_cores': int(input("Number of Processor Cores: ")),
        'pc': int(input("Primary Camera (in MP): ")),
        'px_height': int(input("Pixel Height: ")),
        'px_width': int(input("Pixel Width: ")),
        'ram': int(input("RAM (in MB): ")),
        'sc_h': int(input("Screen Height (in cm): ")),
        'sc_w': int(input("Screen Width (in cm): ")),
        'talk_time': int(input("Talk Time (in hours): ")),
        'three_g': int(input("3G Support (1 = Yes, 0 = No): ")),
        'touch_screen': int(input("Touch Screen (1 = Yes, 0 = No): ")),
        'wifi': int(input("WiFi Support (1 = Yes, 0 = No): "))
    }

    return pd.DataFrame([user_data])

loaded_model = joblib.load("price_predictor_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

user_df = get_user_input()
user_scaled = loaded_scaler.transform(user_df)
prediction = loaded_model.predict(user_scaled)

price_labels = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
print("\nPredicted Price Range:", price_labels[prediction[0]])
