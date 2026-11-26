"""
Backend logic for the AI-Driven Smart Monitoring System.

Everything here comes from the original app.py: SQLite storage, the
Isolation Forest model, severity rules, Telegram alerts, the sensor
simulators, the edge vs cloud-style benchmark, the forecast and the PDF
report. It has been moved into functions so the UI can call it. The
calculations themselves are unchanged.
"""

import sqlite3
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from sklearn.ensemble import IsolationForest


# ==========================================
# CONSTANTS
# ==========================================

DB = "sensor_readings.db"

ROOMS = ["Room 1", "Room 2", "Room 3", "Laboratory"]
SENSOR_MAP = {
    "Room 1": "SENSOR-001",
    "Room 2": "SENSOR-002",
    "Room 3": "SENSOR-003",
    "Laboratory": "SENSOR-004",
}
SEVERITIES = ["Normal", "Warning", "Critical"]
SENSOR_CONDITIONS = [
    "Healthy", "Disconnected", "Missing readings",
    "Frozen / stuck readings", "Abnormal sensor values",
]
DATA_SOURCES = ["Simulated IoT sensor", "Manual test reading"]

# Isolation Forest settings (used by make_model and shown in the UI).
MODEL_CONTAMINATION = 0.08
MODEL_TRAINING_SAMPLES = 500
MODEL_RANDOM_STATE = 42


# ==========================================
# DATABASE
# ==========================================

def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                anomaly INTEGER NOT NULL,
                source TEXT NOT NULL,
                room_id TEXT NOT NULL DEFAULT 'Room 1',
                severity TEXT NOT NULL DEFAULT 'Normal',
                processing_ms REAL NOT NULL DEFAULT 0
            )
        """)
        # Safely upgrade an existing database without deleting saved readings.
        columns = {row[1] for row in con.execute("PRAGMA table_info(readings)")}
        if "room_id" not in columns:
            con.execute("ALTER TABLE readings ADD COLUMN room_id TEXT NOT NULL DEFAULT 'Room 1'")
        if "severity" not in columns:
            con.execute("ALTER TABLE readings ADD COLUMN severity TEXT NOT NULL DEFAULT 'Normal'")
        if "processing_ms" not in columns:
            con.execute("ALTER TABLE readings ADD COLUMN processing_ms REAL NOT NULL DEFAULT 0")
        if "sensor_id" not in columns:
            con.execute("ALTER TABLE readings ADD COLUMN sensor_id TEXT NOT NULL DEFAULT 'SENSOR-001'")
        # Assign stable IDs to older rows using their monitoring location.
        con.execute("""
            UPDATE readings SET sensor_id = CASE room_id
                WHEN 'Room 1' THEN 'SENSOR-001'
                WHEN 'Room 2' THEN 'SENSOR-002'
                WHEN 'Room 3' THEN 'SENSOR-003'
                WHEN 'Laboratory' THEN 'SENSOR-004'
                ELSE 'SENSOR-001' END
            WHERE sensor_id IS NULL OR sensor_id = '' OR sensor_id = 'SENSOR-001'
        """)
        # Keep severity consistent with the current classification rules.
        con.execute("""
            UPDATE readings
            SET severity = CASE
                WHEN COALESCE(anomaly, 0) = 0 THEN 'Normal'
                WHEN temperature >= 40 OR temperature <= 5
                     OR humidity >= 95 OR humidity <= 10 THEN 'Critical'
                ELSE 'Warning'
            END
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS telegram_notifications (
                fingerprint TEXT PRIMARY KEY,
                sent_at TEXT NOT NULL,
                message TEXT NOT NULL
            )
        """)


def save_rows(df):
    with sqlite3.connect(DB) as con:
        df[
            ["timestamp", "temperature", "humidity",
             "anomaly", "source", "room_id", "severity", "processing_ms", "sensor_id"]
        ].to_sql("readings", con, if_exists="append", index=False)


def load_rows():
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query("SELECT * FROM readings ORDER BY id ASC", con)


def load_telegram_log(limit=None):
    query = "SELECT fingerprint, sent_at, message FROM telegram_notifications ORDER BY sent_at DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query(query, con)


def database_stats():
    with sqlite3.connect(DB) as con:
        readings = con.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        alerts = con.execute("SELECT COUNT(*) FROM telegram_notifications").fetchone()[0]
        first_last = con.execute("SELECT MIN(timestamp), MAX(timestamp) FROM readings").fetchone()
    return {"readings": readings, "telegram_sent": alerts,
            "first": first_last[0], "last": first_last[1]}


# ==========================================
# PDF REPORT
# ==========================================

def build_pdf_report(readings, sensor_status, sensor_reason, sensor_action, benchmark=None, forecast=None):
    """Build a downloadable PDF summary from the current prototype data."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm,
        topMargin=16*mm, bottomMargin=16*mm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#17365D")))
    story = [
        Paragraph("AI-Driven Smart Monitoring System", styles["ReportTitle"]),
        Paragraph("UROP Project – Monitoring and Performance Summary", styles["Heading2"]),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M:%S %p')}", styles["Normal"]),
        Spacer(1, 8),
        Paragraph("1. Project Overview", styles["Heading2"]),
        Paragraph("This report summarizes readings and software-prototype results for temperature and humidity monitoring. Sensor inputs are simulated; no physical IoT sensor or real cloud server is connected.", styles["BodyText"]),
        Spacer(1, 8),
    ]

    if readings is None or readings.empty:
        story.append(Paragraph("No sensor readings are currently available.", styles["BodyText"]))
    else:
        total = len(readings)
        anomalies = int(readings["anomaly"].fillna(0).sum())
        normal = total - anomalies
        anomaly_rate = (anomalies / total * 100) if total else 0
        avg_temp = float(readings["temperature"].mean())
        avg_humidity = float(readings["humidity"].mean())
        avg_processing = float(readings["processing_ms"].mean()) if "processing_ms" in readings else 0.0
        summary = [
            ["Metric", "Result"],
            ["Total readings", str(total)],
            ["Normal readings", str(normal)],
            ["AI anomalies", str(anomalies)],
            ["Anomaly rate", f"{anomaly_rate:.1f}%"],
            ["Average temperature", f"{avg_temp:.2f} °C"],
            ["Average humidity", f"{avg_humidity:.2f}%"],
            ["Average local AI processing", f"{avg_processing:.3f} ms"],
        ]
        story += [Paragraph("2. Sensor Data Summary", styles["Heading2"]), _pdf_table(summary), Spacer(1, 8)]

        room_summary = readings.groupby("room_id").agg(
            Readings=("id", "count"), Anomalies=("anomaly", "sum"),
            Avg_Temperature=("temperature", "mean"), Avg_Humidity=("humidity", "mean")
        ).reset_index()
        room_data = [["Room", "Readings", "Anomalies", "Avg temp (°C)", "Avg humidity (%)"]]
        for _, row in room_summary.iterrows():
            room_data.append([str(row["room_id"]), str(int(row["Readings"])), str(int(row["Anomalies"])), f"{row['Avg_Temperature']:.2f}", f"{row['Avg_Humidity']:.2f}"])
        story += [Paragraph("3. Room-wise Summary", styles["Heading2"]), _pdf_table(room_data), Spacer(1, 8)]

        recent = readings.tail(12).sort_values("id", ascending=False)
        recent_data = [["Timestamp", "Room", "Temp °C", "Humidity %", "AI status", "Severity"]]
        for _, row in recent.iterrows():
            recent_data.append([
                str(row.get("timestamp", ""))[:19], str(row.get("room_id", "")),
                f"{float(row['temperature']):.1f}", f"{float(row['humidity']):.1f}",
                "Abnormal" if int(row.get("anomaly", 0)) else "Normal", str(row.get("severity", "Normal"))
            ])
        story += [Paragraph("4. Recent Sensor Readings (up to 12)", styles["Heading2"]), _pdf_table(recent_data, font_size=7), Spacer(1, 8)]

    story += [
        Paragraph("5. Sensor Health Simulation", styles["Heading2"]),
        Paragraph(f"Status: <b>{sensor_status}</b><br/>{sensor_reason}<br/>Recommended action: {sensor_action}", styles["BodyText"]),
        Paragraph("This status is a simulated fault scenario for demonstration, not a diagnosis of physical hardware.", styles["Italic"]),
        Spacer(1, 8),
    ]

    story.append(Paragraph("6. Edge vs Cloud-Style Benchmark", styles["Heading2"]))
    if benchmark:
        bench_rows = [["Metric", "Result"],
            ["Average edge latency", f"{benchmark['edge_average_ms']:.3f} ms"],
            ["Average cloud-style latency", f"{benchmark['cloud_average_ms']:.3f} ms"],
            ["Simulated network delay", f"{benchmark['network_delay_ms']} ms"],
            ["Readings tested", str(benchmark['sample_count'])],
            ["Predictions match", "Yes" if benchmark["predictions_match"] else "No"]]
        story += [_pdf_table(bench_rows), Paragraph("Cloud-style latency includes artificial delay and local model execution; it is not measured real-cloud performance.", styles["Italic"])]
    else:
        story.append(Paragraph("No benchmark has been run in this session. Run the benchmark in the dashboard to include its results.", styles["BodyText"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("7. Temperature & Humidity Forecast", styles["Heading2"]))
    if forecast:
        story.append(Paragraph(f"Forecast room: {forecast['room']}<br/>Historical temperature MAE: {forecast['temp_mae']:.2f} °C<br/>Historical humidity MAE: {forecast['humidity_mae']:.2f}%<br/>Walk-forward test points: {forecast['backtest_count']}", styles["BodyText"]))
        future_rows = [["Forecast time", "Predicted temp (°C)", "Predicted humidity (%)"]]
        for _, row in forecast["future"].iterrows():
            future_rows.append([str(row["Forecast time"]), f"{row['Predicted temperature (°C)']:.2f}", f"{row['Predicted humidity (%)']:.2f}"])
        story.append(_pdf_table(future_rows, font_size=8))
    else:
        story.append(Paragraph("No forecast has been generated in this session. Generate a forecast in the dashboard to include it here.", styles["BodyText"]))

    story += [Spacer(1, 12), Paragraph("8. Prototype Limitations", styles["Heading2"]),
        Paragraph("Sensor readings and hardware-failure scenarios are simulated. Edge processing runs locally on this computer. The cloud-style comparison adds configurable artificial network delay and does not use a deployed cloud service. Forecasts are experimental and should not be used for safety-critical decisions.", styles["BodyText"]),
        Spacer(1, 10), Paragraph("End of Report", styles["Italic"])]
    doc.build(story)
    return buffer.getvalue()


def _pdf_table(rows, font_size=8):
    table = Table(rows, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3F8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


# ==========================================
# TELEGRAM ALERTS
# ==========================================

def get_telegram_credentials():
    """Read bot credentials from Streamlit secrets; never hard-code them."""
    try:
        token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
    except Exception:
        token, chat_id = "", ""
    return str(token).strip(), str(chat_id).strip()


def telegram_configured():
    token, chat_id = get_telegram_credentials()
    return bool(token and chat_id)


def send_telegram_message(message):
    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        return False, "Telegram credentials are not configured."
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("ok"):
            return False, result.get("description", "Telegram API rejected the message.")
        return True, "Telegram message sent."
    except requests.RequestException as exc:
        # Never echo the request URL: it contains the bot token.
        return False, f"Telegram request failed ({type(exc).__name__})."


def alert_fingerprint(alert):
    return "|".join([
        str(alert.get("sensor_id", "")),
        str(alert.get("timestamp", "")),
        str(alert.get("temperature", "")),
        str(alert.get("humidity", "")),
        str(alert.get("severity", "Warning"))
    ])


def notify_telegram_for_rows(rows):
    """Send one Telegram message for each new abnormal reading, without duplicates."""
    if rows is None or rows.empty:
        return
    if not telegram_configured():
        return

    abnormal_rows = rows[rows["anomaly"].astype(int) == 1]
    for _, alert in abnormal_rows.iterrows():
        fingerprint = alert_fingerprint(alert)
        with sqlite3.connect(DB) as con:
            already_sent = con.execute(
                "SELECT 1 FROM telegram_notifications WHERE fingerprint = ?",
                (fingerprint,)
            ).fetchone()
        if already_sent:
            continue

        # Real newlines, so the message reads correctly in Telegram.
        message = (
            "🚨 AI SMART MONITORING ALERT\n"
            f"Severity: {alert.get('severity', 'Warning')}\n"
            f"Location: {alert.get('room_id', 'Unknown')}\n"
            f"Sensor: {alert.get('sensor_id', 'Unknown')}\n"
            f"Temperature: {float(alert['temperature']):.1f} °C\n"
            f"Humidity: {float(alert['humidity']):.1f}%\n"
            f"Time: {alert.get('timestamp', 'Unknown')}\n"
            "AI detected an unusual environmental reading."
        )
        sent, _ = send_telegram_message(message)
        if sent:
            with sqlite3.connect(DB) as con:
                con.execute(
                    "INSERT OR IGNORE INTO telegram_notifications "
                    "(fingerprint, sent_at, message) VALUES (?, ?, ?)",
                    (fingerprint, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)
                )


def attach_delivery_status(alerts):
    """Mark each anomaly row with whether a Telegram alert was delivered for it.

    The match uses sensor, timestamp, temperature and humidity. Severity is
    left out because init_db() can relabel severity on older rows.
    """
    log = load_telegram_log()
    delivered = {}
    for fp, sent_at in zip(log["fingerprint"], log["sent_at"]):
        key = "|".join(str(fp).split("|")[:4])
        delivered.setdefault(key, sent_at)
    out = alerts.copy()
    if out.empty:
        out["Delivered at"] = pd.Series(dtype=object)
        out["Delivery"] = pd.Series(dtype=object)
        return out
    keys = out.apply(lambda r: "|".join(alert_fingerprint(r).split("|")[:4]), axis=1)
    out["Delivered at"] = keys.map(delivered)
    out["Delivery"] = np.where(out["Delivered at"].notna(), "Delivered", "Not sent")
    return out


# ==========================================
# AI MODEL – ISOLATION FOREST
# ==========================================

def make_model():
    rng = np.random.default_rng(MODEL_RANDOM_STATE)
    normal_data = np.column_stack([
        rng.normal(27, 2.2, MODEL_TRAINING_SAMPLES),
        rng.normal(55, 8, MODEL_TRAINING_SAMPLES)
    ])
    model = IsolationForest(contamination=MODEL_CONTAMINATION, random_state=MODEL_RANDOM_STATE)
    model.fit(normal_data)
    return model


@st.cache_resource(show_spinner=False)
def get_model():
    """Train once per server process instead of on every rerun."""
    return make_model()


def classify_severity(temperature, humidity, anomaly):
    """Rule-based severity label layered on top of the AI anomaly result."""
    if not anomaly:
        return "Normal"
    if temperature >= 40 or temperature <= 5 or humidity >= 95 or humidity <= 10:
        return "Critical"
    return "Warning"


def explain_anomaly(row):
    reasons = []
    if row["temperature"] > 35:
        reasons.append("High temperature")
    elif row["temperature"] < 15:
        reasons.append("Low temperature")
    if row["humidity"] > 85:
        reasons.append("High humidity")
    elif row["humidity"] < 25:
        reasons.append("Low humidity")
    if reasons:
        return ", ".join(reasons)
    return "Combination differs from learned normal pattern"


def recommend_action(row):
    """Same rules as the original Smart Recommendations section."""
    reason = explain_anomaly(row)
    if row["severity"] == "Critical":
        return "Critical condition: inspect the area immediately and verify the sensor reading."
    if "temperature" in reason.lower():
        return "Check ventilation, cooling, heating equipment and sensor placement."
    if "humidity" in reason.lower():
        return "Check ventilation, moisture sources and dehumidification/humidification equipment."
    return "Recheck the sensor and compare with nearby monitoring locations."


def add_explanations(df):
    out = df.copy()
    out["AI Status"] = out["anomaly"].map({0: "Normal", 1: "Abnormal"})
    out["Explanation"] = out.apply(
        lambda row: explain_anomaly(row) if row["anomaly"] == 1 else "Within learned normal pattern",
        axis=1
    ) if not out.empty else []
    return out


# ==========================================
# SENSOR HEALTH SIMULATION
# ==========================================

def sensor_health(condition):
    health = {
        "status": "HEALTHY",
        "reason": "Sensor is reporting readings normally in this simulation.",
        "action": "Continue monitoring.",
    }
    if condition == "Disconnected":
        health = {"status": "DISCONNECTED",
                  "reason": "No communication is being received from the simulated sensor.",
                  "action": "Check sensor power, connection and network availability."}
    elif condition == "Missing readings":
        health = {"status": "NO DATA",
                  "reason": "The expected sensor reading is missing.",
                  "action": "Check the sensor and data collection pipeline."}
    elif condition == "Frozen / stuck readings":
        health = {"status": "POSSIBLE SENSOR FAULT",
                  "reason": "The simulated sensor value is assumed to be repeating without changing.",
                  "action": "Inspect the sensor for a stuck value or communication issue."}
    elif condition == "Abnormal sensor values":
        health = {"status": "ABNORMAL VALUES",
                  "reason": "The simulated sensor is reporting values outside the configured safe range.",
                  "action": "Verify the sensor reading and inspect the monitored environment."}
    return health


def health_checks(condition):
    def result(fault):
        if condition == fault:
            return "Failed"
        return "Passed" if condition == "Healthy" else "Simulated fault"
    return pd.DataFrame([
        {"Health check": "Sensor communication", "Result": result("Disconnected")},
        {"Health check": "Reading availability", "Result": result("Missing readings")},
        {"Health check": "Value variation", "Result": result("Frozen / stuck readings")},
        {"Health check": "Value range", "Result": result("Abnormal sensor values")},
    ])


# ==========================================
# SENSOR SIMULATORS
# ==========================================

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def take_live_reading(model, room_id, sensor_id):
    """One automatic simulated reading (original live monitoring panel logic)."""
    rng = np.random.default_rng()
    temperature = float(rng.normal(27, 2.2))
    humidity = float(np.clip(rng.normal(55, 8), 0, 100))

    started = time.perf_counter()
    prediction = model.predict([[temperature, humidity]])[0]
    processing_ms = (time.perf_counter() - started) * 1000
    anomaly = int(prediction == -1)
    severity = classify_severity(temperature, humidity, anomaly)

    row = pd.DataFrame([{
        "timestamp": _now(),
        "temperature": round(temperature, 1),
        "humidity": round(humidity, 1),
        "anomaly": anomaly,
        "source": "Automatic Live Simulation",
        "room_id": room_id,
        "severity": severity,
        "processing_ms": round(processing_ms, 3),
        "sensor_id": sensor_id
    }])
    save_rows(row)
    notify_telegram_for_rows(row)
    return {"temperature": temperature, "humidity": humidity, "anomaly": anomaly,
            "severity": severity, "processing_ms": processing_ms,
            "room_id": room_id, "sensor_id": sensor_id, "timestamp": row.iloc[0]["timestamp"]}


def analyse_manual_reading(model, temp, humidity, source, room_id, sensor_id):
    started = time.perf_counter()
    prediction = model.predict([[temp, humidity]])[0]
    processing_ms = (time.perf_counter() - started) * 1000
    anomaly = int(prediction == -1)
    severity = classify_severity(temp, humidity, anomaly)

    row = pd.DataFrame([{
        "timestamp": _now(),
        "temperature": temp,
        "humidity": humidity,
        "anomaly": anomaly,
        "source": source,
        "room_id": room_id,
        "severity": severity,
        "processing_ms": round(processing_ms, 3),
        "sensor_id": sensor_id
    }])
    save_rows(row)
    notify_telegram_for_rows(row)
    return {"temperature": temp, "humidity": humidity, "anomaly": anomaly,
            "severity": severity, "processing_ms": processing_ms,
            "room_id": room_id, "sensor_id": sensor_id}


def generate_sample_readings(model):
    """30 readings: mostly normal plus three unusual test readings."""
    rng = np.random.default_rng()
    temperatures = rng.normal(27, 2.2, 30)
    humidities = rng.normal(55, 8, 30)
    temperatures[-3:] = [43, 8, 48]
    humidities[-3:] = [92, 15, 98]

    started = time.perf_counter()
    labels = (model.predict(np.column_stack([temperatures, humidities])) == -1).astype(int)
    batch_processing_ms = (time.perf_counter() - started) * 1000
    now = datetime.now()

    rows = pd.DataFrame({
        "timestamp": [(now - pd.Timedelta(minutes=29-i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(30)],
        "temperature": np.round(temperatures, 1),
        "humidity": np.clip(np.round(humidities, 1), 0, 100),
        "anomaly": labels,
        "source": "Simulated IoT sensor",
        "room_id": ROOMS * 7 + ["Room 1", "Room 2"],
        "severity": [classify_severity(float(t), float(h), int(a)) for t, h, a in zip(temperatures, humidities, labels)],
        "processing_ms": round(batch_processing_ms / 30, 3),
        "sensor_id": [SENSOR_MAP[r] for r in ROOMS] * 7 + ["SENSOR-001", "SENSOR-002"]
    })
    save_rows(rows)
    notify_telegram_for_rows(rows)
    return len(rows)


def generate_multi_sensor_readings(model):
    """10 independent readings for each of the four sensors (40 total)."""
    sensor_rooms = {v: k for k, v in SENSOR_MAP.items()}
    rng_multi = np.random.default_rng()
    multi_rows = []
    now_multi = datetime.now()
    for sid, room_name in sensor_rooms.items():
        idx = int(sid[-1]) - 1
        temps = rng_multi.normal(27 + idx * 0.8, 2.0, 10)
        hums = np.clip(rng_multi.normal(55 + idx * 2, 7, 10), 0, 100)
        temps[-1] = [43, 8, 46, 12][idx]
        hums[-1] = [92, 15, 97, 18][idx]
        labels_multi = (model.predict(np.column_stack([temps, hums])) == -1).astype(int)
        for j in range(10):
            proc_start = time.perf_counter()
            model.predict([[float(temps[j]), float(hums[j])]])
            proc_ms = (time.perf_counter() - proc_start) * 1000
            a = int(labels_multi[j])
            multi_rows.append({
                "timestamp": (now_multi - pd.Timedelta(minutes=9-j)).strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": round(float(temps[j]), 1),
                "humidity": round(float(hums[j]), 1),
                "anomaly": a,
                "source": "Multi-sensor simulation",
                "room_id": room_name,
                "severity": classify_severity(float(temps[j]), float(hums[j]), a),
                "processing_ms": round(proc_ms, 3),
                "sensor_id": sid
            })
    multi_df = pd.DataFrame(multi_rows)
    save_rows(multi_df)
    notify_telegram_for_rows(multi_df)
    return len(multi_df)


# ==========================================
# EDGE VS CLOUD-STYLE BENCHMARK
# ==========================================

def run_benchmark(model, samples, network_delay_ms):
    edge_times, cloud_times = [], []
    edge_predictions, cloud_predictions = [], []

    for sample in samples:
        # Edge: local model prediction only.
        edge_start = time.perf_counter()
        edge_prediction = model.predict([sample])[0]
        edge_times.append((time.perf_counter() - edge_start) * 1000)
        edge_predictions.append(int(edge_prediction))

        # Cloud-style: same model plus an explicitly simulated network delay.
        cloud_start = time.perf_counter()
        cloud_prediction = model.predict([sample])[0]
        time.sleep(network_delay_ms / 1000.0)
        cloud_times.append((time.perf_counter() - cloud_start) * 1000)
        cloud_predictions.append(int(cloud_prediction))

    edge_average = float(np.mean(edge_times))
    cloud_average = float(np.mean(cloud_times))
    latency_difference = cloud_average - edge_average
    slower_percent = (latency_difference / edge_average) * 100 if edge_average > 0 else 0.0

    return {
        "edge_average_ms": edge_average,
        "cloud_average_ms": cloud_average,
        "network_delay_ms": network_delay_ms,
        "sample_count": len(samples),
        "predictions_match": edge_predictions == cloud_predictions,
        "latency_difference_ms": latency_difference,
        "slower_percent": slower_percent,
        "edge_times": edge_times,
        "cloud_times": cloud_times,
    }


# ==========================================
# FORECAST (linear trend + walk-forward back-test)
# ==========================================

def prepare_forecast_source(all_rows, room):
    src = all_rows[all_rows["room_id"] == room].copy()
    src["timestamp_dt"] = pd.to_datetime(src["timestamp"], errors="coerce")
    return src.dropna(subset=["timestamp_dt"]).sort_values(["timestamp_dt", "id"]).tail(30)


def compute_forecast(forecast_source, room):
    x_all = (forecast_source["timestamp_dt"] - forecast_source["timestamp_dt"].min()).dt.total_seconds().to_numpy(dtype=float) / 60.0
    future_minutes = np.arange(1, 6, dtype=float)
    x_future = x_all.max() + future_minutes

    def trend_forecast(values, future_x):
        values = np.asarray(values, dtype=float)
        # Fit on at most the 20 most recent points to emphasize recent behaviour.
        fit_x = x_all[-20:]
        fit_y = values[-20:]
        if len(np.unique(fit_x)) < 2:
            return np.full(len(future_x), float(fit_y[-1]))
        slope, intercept = np.polyfit(fit_x, fit_y, 1)
        return intercept + slope * future_x

    temp_values = forecast_source["temperature"].to_numpy(dtype=float)
    humidity_values = forecast_source["humidity"].to_numpy(dtype=float)
    predicted_temp = np.clip(trend_forecast(temp_values, x_future), 0, 60)
    predicted_humidity = np.clip(trend_forecast(humidity_values, x_future), 0, 100)

    # Walk-forward back-test: each historical point is predicted using earlier points only.
    temp_actual_test, temp_pred_test = [], []
    humidity_actual_test, humidity_pred_test = [], []
    for idx in range(5, len(forecast_source)):
        train_start = max(0, idx - 20)
        train_x = x_all[train_start:idx]
        test_x = x_all[idx]
        if len(np.unique(train_x)) < 2:
            continue
        temp_slope, temp_intercept = np.polyfit(train_x, temp_values[train_start:idx], 1)
        humidity_slope, humidity_intercept = np.polyfit(train_x, humidity_values[train_start:idx], 1)
        temp_pred_test.append(float(np.clip(temp_intercept + temp_slope * test_x, 0, 60)))
        humidity_pred_test.append(float(np.clip(humidity_intercept + humidity_slope * test_x, 0, 100)))
        temp_actual_test.append(float(temp_values[idx]))
        humidity_actual_test.append(float(humidity_values[idx]))

    temp_mae = float(np.mean(np.abs(np.array(temp_actual_test) - np.array(temp_pred_test)))) if temp_actual_test else float("nan")
    humidity_mae = float(np.mean(np.abs(np.array(humidity_actual_test) - np.array(humidity_pred_test)))) if humidity_actual_test else float("nan")

    future_times = [forecast_source["timestamp_dt"].max() + pd.Timedelta(minutes=int(m)) for m in future_minutes]
    forecast_table = pd.DataFrame({
        "Forecast time": future_times,
        "Predicted temperature (°C)": np.round(predicted_temp, 2),
        "Predicted humidity (%)": np.round(predicted_humidity, 2)
    })
    return {
        "room": room,
        "history": forecast_source.copy(),
        "future": forecast_table,
        "temp_mae": temp_mae,
        "humidity_mae": humidity_mae,
        "backtest_count": len(temp_actual_test)
    }


def forecast_backtest_series(history):
    """Back-test predictions for each historical point (used by the charts)."""
    hist_x = (history["timestamp_dt"] - history["timestamp_dt"].min()).dt.total_seconds().to_numpy(dtype=float) / 60.0
    temp_hist = history["temperature"].to_numpy(dtype=float)
    humidity_hist = history["humidity"].to_numpy(dtype=float)
    temp_backtest = [None] * len(history)
    humidity_backtest = [None] * len(history)
    for idx in range(5, len(history)):
        train_start = max(0, idx - 20)
        train_x = hist_x[train_start:idx]
        if len(np.unique(train_x)) < 2:
            continue
        ts, ti = np.polyfit(train_x, temp_hist[train_start:idx], 1)
        hs, hi = np.polyfit(train_x, humidity_hist[train_start:idx], 1)
        temp_backtest[idx] = float(np.clip(ti + ts * hist_x[idx], 0, 60))
        humidity_backtest[idx] = float(np.clip(hi + hs * hist_x[idx], 0, 100))
    return temp_hist, humidity_hist, temp_backtest, humidity_backtest
