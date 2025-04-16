

# 🚗 Real-Time Accident Detection System

A machine learning-powered web application to detect real-time accidents and predict severity, along with visualizing accident hot zones on a map. Built with **Streamlit**, **scikit-learn**, and **pydeck**.

## Features
- 🚦 **Accident Detection**: Predict if an accident is likely based on vehicle parameters like speed, acceleration, impact force, and road conditions.
- 🩺 **Severity Prediction**: Estimate the severity of the accident (low, medium, or high) based on the accident's features.
- 🗺️ **Accident Hot Zones**: Visualize simulated accident hot zones on a map using **pydeck**.
- 🧠 **Machine Learning Models**: Trained models to classify accidents and predict severity using a **Random Forest Classifier**.

## Tech Stack
- **Frontend**: Streamlit for the web interface
- **Backend**: scikit-learn for machine learning, pydeck for map visualization
- **Languages**: Python

## Setup and Installation

### Prerequisites
- Python 3.x
- Virtual environment (optional but recommended)

### Steps to Run the Application

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/accident-detection.git
   cd accident-detection
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   On **Windows**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   On **macOS/Linux**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to [http://localhost:8501](http://localhost:8501) to see the application.

## Models Used
- **Accident Classifier**: Predicts whether an accident is likely based on input features.
- **Severity Predictor**: Classifies the severity of the accident as `low`, `medium`, or `high`.

## Input Features
- **Speed**: Vehicle speed in km/h.
- **Acceleration**: Vehicle acceleration in m/s².
- **Impact Force**: Estimated impact force (0-10 scale).
- **Road Condition**: Type of road condition (encoded).

## How it Works
- Enter the vehicle parameters (speed, acceleration, impact force, and road condition) in the sidebar.
- Click on the **"Predict Accident & Severity"** button to get real-time accident predictions and severity.
- The map will display **simulated accident hot zones** based on random data points (this can be replaced with real-time accident data).

## Simulated Data
The application currently uses simulated accident data for the hot zone visualization. In a production environment, this could be integrated with **real-time GPS data** from vehicles or accident reporting systems.

## Contributing

If you'd like to contribute to this project, feel free to fork the repo and create a pull request. You can help with:
- Improving the models
- Adding new features
- Fixing bugs

Please make sure to follow proper coding standards and write tests for your contributions.

## License
This project is licensed under the MIT License.

---
