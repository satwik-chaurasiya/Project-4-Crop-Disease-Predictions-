import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def generate_agricultural_data(n_samples=5000):
    """Generates synthetic environmental and soil data linked to crop health."""
    print("[1/4] Generating environmental sensor data...")
    
    # FIX: Use the modern numpy.random.Generator API instead of legacy functions
    rng = np.random.default_rng(seed=42)
    
    data = {
        'Temperature_C': rng.uniform(15, 40, n_samples),
        'Humidity_Pct': rng.uniform(40, 95, n_samples),
        'Soil_Moisture_Pct': rng.uniform(20, 80, n_samples),
        'Rainfall_mm': rng.uniform(50, 300, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Rule engine to assign logical crop conditions / diseases
    def determine_condition(row):
        if row['Humidity_Pct'] > 80 and 20 <= row['Temperature_C'] <= 28:
            return 'Fungal Blight'
        elif row['Soil_Moisture_Pct'] < 30 and row['Temperature_C'] > 35:
            return 'Root Rot / Dehydration'
        elif row['Rainfall_mm'] > 220 and row['Humidity_Pct'] > 75:
            return 'Bacterial Leaf Streak'
        else:
            return 'Healthy'
            
    df['Crop_Status'] = df.apply(determine_condition, axis=1)
    return df

def train_crop_model(df):
    """Trains a Decision Tree model to classify crop health."""
    print("[2/4] Splitting datasets and training Decision Tree...")
    
    X = df.drop(columns=['Crop_Status'])
    y = df['Crop_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # FIX: Explicitly set ccp_alpha (cost-complexity pruning) to silence SonarQube python:S6973
    model = DecisionTreeClassifier(max_depth=5, random_state=42, ccp_alpha=0.0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"\n>>> Crop Model Prediction Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    
    return model, X.columns

def diagnostic_engine(model, features, temp, humidity, moisture, rainfall):
    """Predicts crop disease and prints an actionable treatment plan."""
    input_data = pd.DataFrame([[temp, humidity, moisture, rainfall]], columns=features)
    prediction = model.predict(input_data)[0]
    
    treatments = {
        'Fungal Blight': "Apply copper-based fungicides immediately. Reduce overhead watering to lower humidity canopy.",
        'Root Rot / Dehydration': "Initiate emergency drip irrigation. Apply organic mulch to retain soil moisture.",
        'Bacterial Leaf Streak': "Prune infected leaves. Avoid nitrogen over-fertilization and ensure proper field drainage.",
        'Healthy': "Environmental conditions are optimal. Continue standard crop rotation and monitoring schedule."
    }
    
    print("\n" + "="*60)
    print("         AGRI-AI: CROP DISEASE DIAGNOSTIC SYSTEM       ")
    print("="*60)
    print(f"Sensors -> Temp: {temp}°C | Humidity: {humidity}% | Soil Moisture: {moisture}% | Rain: {rainfall}mm")
    print(f"AI Diagnostics: {prediction.upper()}")
    print(f"Treatment Plan: {treatments[prediction]}")
    print("="*60 + "\n")

def main():
    df = generate_agricultural_data()
    model, feature_columns = train_crop_model(df)
    
    print("[3/4] Simulating field sensor anomalies...")
    diagnostic_engine(model, feature_columns, temp=24, humidity=88, moisture=50, rainfall=120)
    diagnostic_engine(model, feature_columns, temp=38, humidity=45, moisture=22, rainfall=60)
    
    print("[4/4] Activating Real-Time Field Simulation.")
    while True:
        try:
            flag = input("Press Enter to input sensor logs or type 'exit' to quit: ").strip().lower()
            if flag == 'exit':
                break
            
            t = float(input("Ambient Temperature (°C): "))
            h = float(input("Relative Humidity (%): "))
            m = float(input("Soil Moisture Content (%): "))
            r = float(input("Recent Rainfall Volume (mm): "))
            
            diagnostic_engine(model, feature_columns, t, h, m, r)
        except ValueError:
            print("\nError: Please supply valid numerical data for sensors.\n")

if __name__ == "__main__":
    main()