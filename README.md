# AI-Driven Smart Monitoring System Using IoT & Edge Computing

An AI-powered smart monitoring prototype that analyses temperature and humidity readings, detects abnormal patterns, monitors sensor health and visualises results through a Streamlit dashboard.

## 1. Project Overview

This project demonstrates how Edge Computing and Machine Learning can be combined to monitor environmental sensor data locally.

The system uses simulated sensor readings to demonstrate real-time monitoring, anomaly detection, sensor fault simulation, alert management and performance comparison between local edge processing and a cloud-style processing benchmark.

## 2. Key Features

- **Live Monitoring:** Continuously generates and displays simulated sensor readings.
- **Multi-Sensor Monitoring:** Monitors four virtual sensors across different locations.
- **AI Anomaly Detection:** Uses the Isolation Forest algorithm to identify unusual readings.
- **Sensor Health Monitoring:** Simulates sensor fault conditions and evaluates sensor readings.
- **Edge vs Cloud-Style Benchmark:** Compares local processing latency with a cloud-style benchmark using artificial network delay.
- **Forecasting:** Generates experimental temperature and humidity forecasts.
- **Alert Management:** Maintains warning and critical alert history.
- **Telegram Notifications:** Supports Telegram alerts for detected events.
- **Data Visualisation:** Displays interactive charts and sensor statistics using Plotly.
- **Database Storage:** Stores sensor readings and notification records using SQLite.
- **Reports and Exports:** Supports CSV data exports and PDF report generation.

## 3. Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Anomaly Detection | Isolation Forest |
| Visualisation | Plotly |
| Database | SQLite |
| Notifications | Telegram Bot API |
| Report Generation | ReportLab |

## 4. System Architecture

1. **IoT Layer:** Generates simulated temperature and humidity readings.
2. **Edge Processing Layer:** Processes readings locally.
3. **AI Layer:** Uses Isolation Forest to detect anomalies.
4. **Database Layer:** Stores readings and alert history in SQLite.
5. **Visualisation Layer:** Displays monitoring results through a Streamlit dashboard.

## 5. Project Structure

```text
AI_Smart_Monitoring_UROP/
│
├── app.py
├── backend.py
├── ui.py
├── requirements.txt
├── README.md
├── sensor_readings.db
└── .streamlit/
```

The database and local secrets are excluded from the GitHub repository.

## 6. Installation and Execution

### Requirements
- Python 3.10 or newer
- pip
- Internet connection for installing dependencies

### Installation

```bash
python -m pip install -r requirements.txt
```

### Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## 7. Experimental Results

During prototype testing:

| Metric | Result |
|---|---:|
| Total Sensor Readings | 94 |
| Normal Readings | 80 |
| Detected Anomalies | 14 |
| Anomaly Rate | 14.9% |
| Average Edge Latency | 20.543 ms |
| Average Cloud-Style Latency | 101.904 ms |
| Artificial Delay | 80 ms |
| Forecast Temperature MAE | 4.25°C |
| Forecast Humidity MAE | 14.43% |

The benchmark demonstrated lower processing latency in the local edge-style setup.

**Note:** These are experimental prototype results, not guaranteed production performance. The cloud-style benchmark uses artificial delay and is not a measurement of an actual deployed cloud service.

## 8. Limitations

- No physical sensors are currently connected.
- Temperature and humidity readings are simulated.
- Sensor fault scenarios are simulated.
- Edge processing runs locally on the laptop.
- Cloud-style latency is simulated using artificial delay.
- Forecasting is experimental and is not intended for safety-critical decisions.

## 9. Future Enhancements

- Integrate physical sensors using ESP32 or Raspberry Pi.
- Implement real-time MQTT-based sensor communication.
- Deploy edge processing on dedicated hardware.
- Integrate an actual cloud platform.
- Improve forecasting accuracy using larger datasets.
- Add authentication and role-based access.
- Extend monitoring to additional environmental parameters.

## 10. Project Objective

The main objective is to demonstrate how AI-based anomaly detection and Edge Computing can support faster local analysis, environmental monitoring and early identification of unusual sensor behaviour.
