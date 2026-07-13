# 🚦 Intelligent Traffic Management using AI

---
## 📌 Project Description
This repository contains an end-to-end implementation of an **Intelligent Traffic Management System** driven by machine learning. Traditional traffic light grids operate on static timers, which leads to unnecessary delays, congestion, and increased carbon emissions during peak hours. 

This project solves that inefficiency by utilizing a **Random Forest Classifier** trained on dynamic environmental and intersection data. The system accurately predicts real-time traffic congestion tiers ("Light", "Moderate", "Heavy") and passes those outputs to an adaptive rules engine that optimizes green light signal phases dynamically.

---

## 🎯 Project Objective
* **Minimize Intersection Congestion:** Replace outdated, fixed-cycle traffic signal timers with data-driven dynamic phase allocation.
* **Optimize Traffic Throughput:** Ensure heavy gridlock situations receive priority green light timing to actively flush lanes.
* **Predictive Response:** Factor in secondary constraints like bad weather conditions (rain, fog) and rush-hour schedules to anticipate bottleneck accumulation before it happens.

---

## ✨ Features
* **ML-Powered Diagnostics:** Uses a robust Random Forest pipeline to classify traffic densities with high accuracy.
* **Dynamic Signal Control Engine:** Programmatic logic mapping that translates AI inference directly into actionable signal extensions (+15s to +30s) or contractions (-10s).
* **Synthetic Data Simulator:** Generates complex intersection data variants featuring hour blocks, day tracking, vehicle speeds, and weather states.
* **Interactive Sandbox CLI:** An embedded command-line interface allowing users to manually pass custom intersection metrics to observe system decisions instantly.

---

## 🛠️ System Architecture & Workflow



1. **Sensor Ingestion Simulation:** Collects current time, day, weather status, vehicle count, and average lane speeds.
2. **Preprocessing Pipeline:** Normalizes features using `StandardScaler` and encodes weather states via `LabelEncoder`.
3. **AI Inference:** The trained Random Forest model predicts the current intersection stress level.
4. **Signal Adjuster:** Calculates optimal green light duration and prints actionable diagnostics to the dashboard.

---

## 🚀 Method to Use

### 1. Prerequisites & Installation
Ensure you have Python installed, then clone the repository and install dependencies:
```text
│
├──pip install numpy pandas scikit-learn matplotlib seaborn
│
├──Follow the command line prompts.
│
├──Provide variables like Hour of Day (0-23), Vehicle Count, and Weather (Clear, Rainy, Foggy).
│
├──Review the printed optimization dashboard showing the new green light duration.
│
└── Type exit to cleanly terminate the program. 
```

---
# 🌾 AI-Based Crop Disease Prediction System

---
## 📌 Project Description
Built for modern Smart Agriculture (Agri-Tech) frameworks, this repository hosts an **AI-Based Crop Disease Prediction System**. Environmental irregularities drastically compromise crop yields worldwide. 

This project tackles this challenge by deploying an optimized **Decision Tree Classifier** that analyzes multi-channel environmental sensor streams—ambient temperature, relative humidity, soil moisture levels, and recent rainfall volumes. It acts as a digital agronomist, diagnosing specific abnormalities like *Fungal Blight*, *Root Rot*, or *Bacterial Leaf Streak*, while serving prescriptive treatment actions immediately.

---

## 🎯 Project Objective
* **Early Anomaly Detection:** Classify underlying vegetation stress before physiological damage spreads across fields.
* **Prescriptive Agri-Intelligence:** Link machine learning classification outputs with instantly actionable chemical, structural, or irrigation treatment solutions.
* **SonarQube Quality Standards:** Formulated strictly under modern clean-code standards, completely eliminating legacy random state functions and optimizing tree depth against overfitting.

---

## ✨ Features
* **Explainable AI Architecture:** Uses a optimized Decision Tree framework, mirroring natural agronomic diagnostic paths for full transparency.
* **Prescriptive Treatment Engine:** Features built-in action mapping, serving operational guidelines (e.g., *fungicide choice, drainage fixes, irrigation shifts*) alongside disease predictions.
* **Clean & Modern Codebase:** Fully compliant with modern code practices, implementing `numpy.random.default_rng` structures and explicit `ccp_alpha` variables to ensure robust quality.
* **Real-Time CLI Input Terminal:** Simulates live field data logs via an interactive user loop.

---

## 🔬 Core Environmental Vectors & Logic



The predictive model constructs split conditions across these sensor profiles:
* **Fungal Blight:** Triggered by high humidity vectors ($>80\%$) held at moderate temperature windows ($20^\circ\text{C} - 28^\circ\text{C}$).
* **Root Rot / Dehydration:** Identified by low moisture indices ($<30\%$) coupled with excessive ground heat ($>35^\circ\text{C}$).
* **Bacterial Leaf Streak:** Pinpointed by high rainfall thresholds ($>220\text{mm}$) combined with elevated humidity.

---

## 🚀 Method to Use
---
Clone the repository and install the standard machine learning libraries:
```text
│
├──pip install numpy pandas scikit-learn
│
└── Enter the values correctly
```
---