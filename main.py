"""
Session 18 Final Project: End-to-End IoT Data Pipeline
Institute: IASBS

Project Overview:
1. Data Acquisition (real + synthetic sensor data)
2. Data Cleaning & Preprocessing
3. ML Model Training & ONNX Export
4. Model Loading & Prediction
5. Evaluation & Visualization
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

import os
import shutil
import kagglehub

import warnings

warnings.filterwarnings('ignore')

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    print("Warning: ONNX libraries not installed. Install with: pip install skl2onnx onnxruntime")
    ONNX_AVAILABLE = False


class IoTDataPipeline:
    """
    Main class for IoT sensor data pipeline using OOP principles
    """

    def __init__(self, project_name="IoT_Sensor_Project"):
        self.project_name = project_name
        self.raw_data = None
        self.clean_data = None
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.onnx_model_path = "temperature_model.onnx"
        self.data_source = None

        print(f"=== {self.project_name} Initialized ===")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ========== STEP 1: تولید دیتا ==========
    def generate_synthetic_data(self, n_samples=500, save_csv=True):
        """
        Generate synthetic IoT sensor data with realistic patterns
        Includes: Temperature, Humidity, Pressure, with noise and missing values
        """
        print("=" * 50)
        print("STEP 1: DATA ACQUISITION (SYNTHETIC DATA)")
        print("=" * 50)

        time = pd.date_range("2025-01-01", periods=n_samples, freq="h")
        hours = np.arange(n_samples)

        temperature = 20 + 8 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 1.5, n_samples)

        humidity = 60 - 0.5 * temperature + 10 * np.cos(2 * np.pi * hours / 24) + np.random.normal(0, 3, n_samples)
        humidity = np.clip(humidity, 20, 100)

        pressure = 1013 + 5 * np.sin(2 * np.pi * hours / 168) + np.random.normal(0, 2, n_samples)

        self.raw_data = pd.DataFrame({
            "Timestamp": time,
            "Temperature": temperature,
            "Humidity": humidity,
            "Pressure": pressure
        })

        missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
        self.raw_data.loc[missing_indices, 'Temperature'] = np.nan

        outlier_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
        self.raw_data.loc[outlier_indices, 'Temperature'] += np.random.choice([-15, 15], size=len(outlier_indices))

        self.data_source = "synthetic"

        if save_csv:
            self.raw_data.to_csv("sensor_data_raw.csv", index=False)
            print(f"✓ Generated {n_samples} sensor readings")
            print(f"✓ Saved to: sensor_data_raw.csv")

        print(f"\nDataset Info:")
        print(f"  - Total samples: {len(self.raw_data)}")
        print(f"  - Missing values: {self.raw_data['Temperature'].isna().sum()}")
        print(f"  - Features: {list(self.raw_data.columns)}")
        print(f"\nFirst 5 rows:")
        print(self.raw_data.head())
        print()

        return self.raw_data

    def import_real_data(self, save_csv=True):
        """
        Import real weather data from KaggleHub and standardize column names
        """
        print("=" * 50)
        print("STEP 1: DATA ACQUISITION (REAL DATA)")
        print("=" * 50)

        cache_path = kagglehub.dataset_download("muthuj7/weather-dataset")
        print("✓ Dataset downloaded to cache")

        PROJECT_PATH = r"C:/Users/Samin/PycharmProjects/pythonProject7"
        CSV_NAME = "weatherHistory.csv"

        os.makedirs(PROJECT_PATH, exist_ok=True)

        src = os.path.join(cache_path, CSV_NAME)
        dst = os.path.join(PROJECT_PATH, CSV_NAME)

        shutil.copy2(src, dst)
        print(f"✓ CSV copied to project folder")

        df_raw = pd.read_csv(dst)

        column_mapping = {
            'Formatted Date': 'Timestamp',
            'Temperature (C)': 'Temperature',
            'Pressure (millibars)': 'Pressure'
        }

        df_raw = df_raw.rename(columns=column_mapping)

        self.raw_data = df_raw[['Timestamp', 'Temperature', 'Humidity', 'Pressure']].copy()

        self.data_source = "kaggle"

        if save_csv:
            self.raw_data.to_csv("sensor_data_raw.csv", index=False)
            print(f"✓ Standardized data saved to: sensor_data_raw.csv")

        print(f"\nDataset Info:")
        print(f"  - Total samples: {len(self.raw_data)}")
        print(f"  - Missing values per column:")
        print(self.raw_data.isna().sum())
        print(f"  - Features: {list(self.raw_data.columns)}")

        print(f"\nFirst 5 rows:")
        print(self.raw_data.head())
        print()

        return self.raw_data

    # ========== STEP 2: DATA CLEANING & PREPROCESSING ==========
    def clean_and_preprocess(self):
        """
        Clean sensor data: handle missing values, remove outliers, normalize
        """
        print("=" * 50)
        print("STEP 2: DATA CLEANING & PREPROCESSING")
        print("=" * 50)

        if self.raw_data is None:
            raise ValueError("No raw data available. Run generate_synthetic_data() or import_real_data() first.")

        self.clean_data = self.raw_data.copy()

        # 1. هندل قسمت های گم شده
        missing_before = self.clean_data['Temperature'].isna().sum()
        self.clean_data['Temperature'].fillna(method='ffill', inplace=True)
        self.clean_data['Temperature'].fillna(method='bfill', inplace=True)
        print(f"✓ Filled {missing_before} missing temperature values")

        # 2. Remove outliers using IQR method
        Q1 = self.clean_data['Temperature'].quantile(0.25)
        Q3 = self.clean_data['Temperature'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_before = len(self.clean_data)
        self.clean_data = self.clean_data[
            (self.clean_data['Temperature'] >= lower_bound) &
            (self.clean_data['Temperature'] <= upper_bound)
            ]
        outliers_removed = outliers_before - len(self.clean_data)
        print(f"✓ Removed {outliers_removed} outliers (IQR method)")

        # 3. Apply moving average smoothing
        window_size = 5
        self.clean_data['Temperature_Smooth'] = self.clean_data['Temperature'].rolling(
            window=window_size, center=True
        ).mean()
        self.clean_data['Temperature_Smooth'].fillna(
            self.clean_data['Temperature'], inplace=True
        )
        print(f"✓ Applied moving average filter (window={window_size})")

        # 4. Feature engineering: create time-based features
        self.clean_data['Timestamp'] = pd.to_datetime(
            self.clean_data['Timestamp'],
            utc=True,
            errors='coerce'
        )

        self.clean_data['Hour'] = pd.to_datetime(self.clean_data['Timestamp']).dt.hour
        self.clean_data['Day'] = pd.to_datetime(self.clean_data['Timestamp']).dt.day
        print(f"✓ Created time-based features")

        print(f"\nCleaned dataset shape: {self.clean_data.shape}")
        print(f"Statistics after cleaning:")
        print(self.clean_data[['Temperature', 'Humidity', 'Pressure']].describe())
        print()

        return self.clean_data

    # ========== STEP 3: MODEL TRAINING & ONNX EXPORT ==========
    def train_model_and_export_onnx(self, target='Temperature'):
        """
        Train ML model to predict temperature based on other features
        Export trained model to ONNX format
        """
        print("=" * 50)
        print("STEP 3: MODEL TRAINING & ONNX EXPORT")
        print("=" * 50)

        if self.clean_data is None:
            raise ValueError("No clean data available. Run clean_and_preprocess() first.")

        feature_cols = ['Humidity', 'Pressure', 'Hour']
        X = self.clean_data[feature_cols].values
        y = self.clean_data[target].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        print(f"✓ Features used: {feature_cols}")
        print(f"✓ Target variable: {target}")
        print(f"✓ Training samples: {len(self.X_train)}")
        print(f"✓ Testing samples: {len(self.X_test)}")

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"✓ Applied StandardScaler normalization")

        self.model = LinearRegression()
        self.model.fit(self.X_train_scaled, self.y_train)

        train_score = self.model.score(self.X_train_scaled, self.y_train)
        print(f"✓ Model trained successfully")
        print(f"  Training R² Score: {train_score:.4f}")

        if ONNX_AVAILABLE:
            try:
                initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
                onnx_model = convert_sklearn(
                    self.model,
                    initial_types=initial_type,
                    target_opset=12
                )

                with open(self.onnx_model_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())

                print(f"✓ Model exported to ONNX: {self.onnx_model_path}")
            except Exception as e:
                print(f"✗ ONNX export failed: {e}")
        else:
            print("✗ ONNX export skipped (libraries not available)")

        print()
        return self.model

    # ========== STEP 4: LOAD & PREDICT ==========
    def load_onnx_and_predict(self):
        """
        Load ONNX model and make predictions on test data
        """
        print("=" * 50)
        print("STEP 4: LOAD ONNX MODEL & PREDICT")
        print("=" * 50)

        if not ONNX_AVAILABLE:
            print("Using scikit-learn model for predictions (ONNX not available)")
            self.predictions = self.model.predict(self.X_test_scaled)
            print(f"✓ Generated {len(self.predictions)} predictions")
            print()
            return self.predictions

        try:
            ort_session = ort.InferenceSession(self.onnx_model_path)
            print(f"✓ Loaded ONNX model from: {self.onnx_model_path}")

            input_name = ort_session.get_inputs()[0].name
            self.predictions = ort_session.run(
                None,
                {input_name: self.X_test_scaled.astype(np.float32)}
            )[0]

            print(f"✓ Generated {len(self.predictions)} predictions using ONNX")
            print(f"\nSample predictions (first 5):")
            for i in range(min(5, len(self.predictions))):
                print(f"  Actual: {self.y_test[i]:.2f}°C | Predicted: {self.predictions[i]:.2f}°C")

        except Exception as e:
            print(f"✗ ONNX prediction failed: {e}")
            print("Falling back to scikit-learn model...")
            self.predictions = self.model.predict(self.X_test_scaled)
            print(f"✓ Generated {len(self.predictions)} predictions")

        print()
        return self.predictions

    # ========== STEP 5: EVALUATION & VISUALIZATION ==========
    def evaluate_and_visualize(self):
        """
        Evaluate model performance and create visualizations
        """
        print("=" * 50)
        print("STEP 5: EVALUATION & VISUALIZATION")
        print("=" * 50)

        if self.predictions is None:
            raise ValueError("No predictions available. Run load_onnx_and_predict() first.")

        # Calculate metrics
        mse = mean_squared_error(self.y_test, self.predictions)
        mae = mean_absolute_error(self.y_test, self.predictions)
        r2 = r2_score(self.y_test, self.predictions)
        rmse = np.sqrt(mse)

        print("Model Performance Metrics:")
        print(f"  • Mean Squared Error (MSE): {mse:.4f}")
        print(f"  • Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  • Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  • R² Score: {r2:.4f}")
        print()

        # Create visualizations
        self._create_visualizations()

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def _create_visualizations(self):
        """
        Generate comprehensive visualization plots
        """
        fig = plt.figure(figsize=(16, 10))

        # Plot 1: Actual vs Predicted
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(self.y_test, self.predictions, alpha=0.6, color='blue', edgecolors='black')
        ax1.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Temperature (°C)', fontsize=11)
        ax1.set_ylabel('Predicted Temperature (°C)', fontsize=11)
        ax1.set_title('Actual vs Predicted Temperature', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residuals
        ax2 = plt.subplot(2, 3, 2)
        residuals = self.y_test - self.predictions
        ax2.scatter(self.predictions, residuals, alpha=0.6, color='green', edgecolors='black')
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Temperature (°C)', fontsize=11)
        ax2.set_ylabel('Residuals', fontsize=11)
        ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Prediction Error Distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(residuals, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Prediction Error (°C)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Time Series - Raw vs Clean Temperature
        ax4 = plt.subplot(2, 3, 4)
        sample_size = min(200, len(self.clean_data))
        ax4.plot(self.clean_data['Temperature'][:sample_size],
                 label='Raw Temperature', alpha=0.6, linewidth=1)
        ax4.plot(self.clean_data['Temperature_Smooth'][:sample_size],
                 label='Smoothed Temperature', linewidth=2, color='red')
        ax4.set_xlabel('Time (hours)', fontsize=11)
        ax4.set_ylabel('Temperature (°C)', fontsize=11)
        ax4.set_title('Temperature Time Series (First 200 Hours)', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Plot 5: Feature Correlation Heatmap (simplified)
        ax5 = plt.subplot(2, 3, 5)
        feature_data = self.clean_data[['Temperature', 'Humidity', 'Pressure', 'Hour']].corr()
        im = ax5.imshow(feature_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax5.set_xticks(range(len(feature_data.columns)))
        ax5.set_yticks(range(len(feature_data.columns)))
        ax5.set_xticklabels(feature_data.columns, rotation=45, ha='right')
        ax5.set_yticklabels(feature_data.columns)
        ax5.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

        # Add correlation values
        for i in range(len(feature_data.columns)):
            for j in range(len(feature_data.columns)):
                text = ax5.text(j, i, f'{feature_data.iloc[i, j]:.2f}',
                                ha="center", va="center", color="black", fontsize=9)

        plt.colorbar(im, ax=ax5)

        # Plot 6: Prediction Timeline Sample
        ax6 = plt.subplot(2, 3, 6)
        sample_indices = np.arange(min(50, len(self.y_test)))
        ax6.plot(sample_indices, self.y_test[:len(sample_indices)],
                 'o-', label='Actual', linewidth=2, markersize=5)
        ax6.plot(sample_indices, self.predictions[:len(sample_indices)],
                 's-', label='Predicted', linewidth=2, markersize=5, alpha=0.7)
        ax6.set_xlabel('Sample Index', fontsize=11)
        ax6.set_ylabel('Temperature (°C)', fontsize=11)
        ax6.set_title('Prediction Timeline (First 50 Samples)', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.suptitle(f'{self.project_name} - Model Evaluation Dashboard',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
        print("✓ Visualizations saved to: model_evaluation_plots.png")
        plt.show()

    def generate_report(self, metrics):
        """
        Generate a summary report of the entire pipeline
        """
        print("\n" + "=" * 50)
        print("FINAL PROJECT REPORT")
        print("=" * 50)

        data_source_info = "Synthetic (Generated)" if self.data_source == "synthetic" else "Real (Kaggle Weather Dataset)"

        report = f"""
PROJECT: {self.project_name}
DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET SUMMARY:
  • Data Source: {data_source_info}
  • Total samples collected: {len(self.raw_data)}
  • Samples after cleaning: {len(self.clean_data)}
  • Features used: Humidity, Pressure, Hour
  • Target variable: Temperature

MODEL PERFORMANCE:
  • Algorithm: Linear Regression
  • R² Score: {metrics['r2']:.4f}
  • RMSE: {metrics['rmse']:.4f}°C
  • MAE: {metrics['mae']:.4f}°C

DELIVERABLES:
  ✓ sensor_data_raw.csv (raw dataset)
  ✓ {self.onnx_model_path} (ONNX model)
  ✓ model_evaluation_plots.png (visualizations)
  ✓ This report

CONCLUSION:
The IoT data pipeline successfully demonstrated end-to-end workflow
from data acquisition through model deployment. The model achieved
an R² score of {metrics['r2']:.4f}, indicating {"good" if metrics['r2'] > 0.8 else "moderate"} predictive performance
for temperature estimation based on humidity, pressure, and time.
"""

        print(report)

        # Save report to file
        with open('project_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

        print("✓ Report saved to: project_report.txt")
        print("=" * 50)
        print("\n PROJECT COMPLETED SUCCESSFULLY! \n")


# ========== MAIN EXECUTION ==========
def main():
    """
    Execute the complete IoT data pipeline
    """
    # Initialize pipeline
    pipeline = IoTDataPipeline(project_name="Temperature_Prediction_System")

    # Step 1: Choose data source
    print("Select data source:")
    print("1. Generate synthetic sensor data")
    print("2. Import real data from Kaggle")

    choose_generate_model = input("\nEnter your choice (1 or 2): ").strip()

    if choose_generate_model == "1":
        pipeline.generate_synthetic_data(n_samples=500, save_csv=True)
    elif choose_generate_model == "2":
        pipeline.import_real_data()
    else:
        print("Invalid choice. Defaulting to synthetic data generation.")
        pipeline.generate_synthetic_data(n_samples=500, save_csv=True)

    # Step 2: Clean and preprocess data
    pipeline.clean_and_preprocess()

    # Step 3: Train model and export to ONNX
    pipeline.train_model_and_export_onnx(target='Temperature')

    # Step 4: Load ONNX model and predict
    pipeline.load_onnx_and_predict()

    # Step 5: Evaluate and visualize results
    metrics = pipeline.evaluate_and_visualize()

    # Generate final report
    pipeline.generate_report(metrics)


if __name__ == "__main__":
    main()
