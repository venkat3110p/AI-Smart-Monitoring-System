# AI-Driven Smart Monitoring System Using IoT & Edge Computing

## Run the prototype
1. Install Python 3.10 or newer.
2. Extract this folder.
3. Open a terminal in this folder.
4. Run:
   `python -m pip install -r requirements.txt`
5. Start the dashboard:
   `python -m streamlit run app.py`
6. A browser tab will open. Use the sidebar to generate sample readings or analyse a reading.

## What is implemented
- Simulated temperature and humidity sensor input
- Local processing on the laptop (edge-style processing)
- Isolation Forest anomaly detection
- Streamlit dashboard and Plotly graph
- SQLite storage of readings

## Current limitation
No physical sensor is connected. Sensor values are simulated. Later, the input function can be replaced with ESP32/Raspberry Pi readings.

## Suggested first demo
Click **Generate 30 sample readings**. The app inserts mostly normal readings and a few unusual test readings. Then show the metrics, chart, AI status, and stored results.
