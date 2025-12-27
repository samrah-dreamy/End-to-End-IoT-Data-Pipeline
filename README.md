# 🌡️ Temperature Prediction System (IoT Data Pipeline)

End-to-end IoT data pipeline for temperature prediction using synthetic or real-world weather data.  
This project demonstrates **data acquisition, preprocessing, ML training, ONNX deployment, evaluation, and reporting** in a clean OOP-based architecture.

---

## 🚀 Project Highlights

- Synthetic & real IoT data support  
- Robust data cleaning (missing values + outliers)  
- Feature engineering on time-series data  
- Linear Regression model (scikit-learn)  
- ONNX export for deploy-ready inference  
- Full evaluation dashboard + final report  
- Clean, explainable, university-ready pipeline  

---

## 📦 Imports – Tools Used in This Project

```python
import pandas as pd
import numpy as np
import matplotlib
````

### Libraries Overview

* **pandas** → Tabular data handling (CSV, DataFrame, cleaning)
* **numpy** → Numerical computations, noise generation, sine waves, randomness
* **matplotlib** → Data visualization (backend defined explicitly)

---

### 🎨 Matplotlib Backend Configuration

```python
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
```

* **TkAgg** uses Tkinter for GUI rendering
* Prevents backend-related issues (especially on Windows & PyCharm)
* `plt` is the main plotting API

📌 *Documentation note:*

> Backend selection ensures GUI compatibility across environments.

---

## 🧠 Machine Learning Utilities

```python
from sklearn.model_selection import train_test_split
```

* Splits data into **train / test**
* Prevents overfitting
* Standard ML pipeline practice

```python
from sklearn.linear_model import LinearRegression
```

* Core ML model
* Assumes a linear relationship between inputs and output

```python
from sklearn.preprocessing import StandardScaler
```

* Feature normalization:

  * Mean = 0
  * Standard deviation = 1
* Essential for linear models & ONNX compatibility

---

### 📏 Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
```

* **MSE** → Penalizes large errors aggressively
* **MAE** → Average absolute error
* **R²** → How well the model explains the data

---

### ⏱️ Utilities

```python
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
```

* `datetime` → Professional timestamps for logs & reports
* `warnings.ignore` → Clean output (perfect for academic projects)

---

## 🔥 ONNX – Deploy-Ready Layer

```python
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as ort
    ONNX_AVAILABLE = True
```

* Converts sklearn model → ONNX
* ONNX model runs **independent of Python & sklearn**

```python
except ImportError:
    ONNX_AVAILABLE = False
```

📌 Graceful failure:
The pipeline does **not crash** if ONNX libraries are missing.
This is a **professional design choice**.

---

## 🧩 Core Architecture – OOP Pipeline

```python
class IoTDataPipeline:
```

* Entire project wrapped in a clean abstraction
* Each pipeline step = one method
* Highly readable, extensible, and reviewer-friendly

---

## ❤️ `__init__` – Pipeline State Initialization

* Project configuration
* Dataset states
* Model & scaler setup
* Logging & timestamps

```python
self.raw_data = None      # Raw dataset
self.clean_data = None    # Cleaned dataset
```

Clear separation ensures **reproducibility**.

```python
self.model = None
self.scaler = StandardScaler()
```

Scaler is shared between train & test (correct ML practice).

---

## 🧪 Step 1 – Synthetic Data Generation

```python
def generate_synthetic_data(self, n_samples=500):
```

### Why synthetic data?

* Fully controllable
* Reproducible
* Perfect for pipeline testing

### Features Simulated

* **Timestamp** → Hourly time-series
* **Temperature** → Daily sinusoidal pattern + noise
* **Humidity** → Inversely correlated with temperature
* **Pressure** → Slow atmospheric variation

### Realism Add-ons

* Missing values → Sensor failure simulation
* Outliers → Sudden sensor spikes
* CSV export for transparency

---

## 🧹 Step 2 – Cleaning & Preprocessing

### Missing Values

```python
fillna(method='ffill')
fillna(method='bfill')
```

* Forward fill → use previous value
* Backward fill → handle edge cases
* No data loss

---

### Outlier Removal (IQR Method)

```python
IQR = Q3 - Q1
```

* Standard statistical approach
* Distribution-independent
* Easy to explain in reports

---

### Smoothing

```python
rolling(window=5, center=True).mean()
```

* Noise reduction
* Trend preservation
* Inspired by signal processing

---

### Feature Engineering ⏱️

```python
Hour = Timestamp.dt.hour
Day  = Timestamp.dt.day
```

* Converts time into ML-friendly numerical features

📌 *Documentation note:*

> Temporal features were extracted to capture daily behavioral patterns.

---

## 🤖 Step 3 – Training & ONNX Export

* Features: `Humidity`, `Pressure`, `Hour`
* Target: `Temperature`

```python
train_test_split(..., random_state=42)
```

* Reproducible
* Scientifically standard

```python
scaler.fit_transform(X_train)
```

* Scaler fits **only on training data**
* Prevents **data leakage** 🔥

```python
convert_sklearn(...)
```

* Produces a deployable ONNX model
* Ready for edge, mobile, or C++ inference

---

## ⚡ Step 4 – Load & Predict

* If ONNX available → `onnxruntime`
* Else → sklearn fallback

📌 Same model, different runtime.

---

## 📊 Step 5 – Evaluation & Visualization

### Metrics Reported

* MSE
* RMSE
* MAE
* R²

### Visualization Dashboard (6 plots)

1. Actual vs Predicted
2. Residuals
3. Error distribution
4. Raw vs smoothed temperature
5. Correlation heatmap
6. Prediction timeline

This section makes the project **feel real & professional**.

---

## 📝 Final Report Generator

```python
generate_report()
```

Includes:

* Dataset summary
* Model configuration
* Performance metrics
* Deliverables list
* Final conclusion

Fully ready for **university submission or company demo**.

---

## 🎬 `main()` – Orchestrator

* Entry point of the program
* Controls execution order
* Handles user input
* Standard Python pattern

---

## ✅ Deliverables

* `sensor_data_raw.csv` – Raw dataset
* `temperature_model.onnx` – Deployable model
* `model_evaluation_plots.png` – Visualization dashboard
* `project_report.txt` – Final report

---

## 🏁 Conclusion

This project demonstrates a complete, clean, and deploy-ready IoT ML pipeline following industry and academic best practices.

---

🔥 **Built for learning, explaining, and shipping.**
