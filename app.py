import sqlite3
import time
import requests
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


# ==========================================
# 1. CONFIGURATION
# ==========================================

DB = "sensor_readings.db"

st.set_page_config(
    page_title="AI Smart Monitoring",
    page_icon="📡",
    layout="wide"
)


# ==========================================
# 2. DATABASE FUNCTIONS
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
        # Repair severity labels on older saved rows. Earlier database versions
        # could assign the default "Normal" label to an AI-detected anomaly.
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
        ].to_sql(
            "readings",
            con,
            if_exists="append",
            index=False
        )


def load_rows():
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query(
            "SELECT * FROM readings ORDER BY id ASC",
            con
        )


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
# TELEGRAM ALERT INTEGRATION
# ==========================================

def get_telegram_credentials():
    """Read bot credentials from Streamlit secrets; never hard-code them."""
    try:
        token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
    except Exception:
        token, chat_id = "", ""
    return str(token).strip(), str(chat_id).strip()


def send_telegram_message(message):
    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        return False, "Telegram credentials are not configured."
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={
    "chat_id": chat_id,
    "text": message,
    "parse_mode": "Markdown"
},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("ok"):
            return False, result.get("description", "Telegram API rejected the message.")
        return True, "Telegram message sent."
    except requests.RequestException as exc:
        return False, f"Telegram request failed: {exc}"

def notify_telegram_for_rows(rows):
    """Send one Telegram message for each new abnormal reading."""
    if rows is None or rows.empty:
        return

    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        return

    abnormal_rows = rows[rows["anomaly"].astype(int) == 1]

    for _, alert in abnormal_rows.iterrows():

        fingerprint = "|".join([
            str(alert.get("sensor_id", "")),
            str(alert.get("timestamp", "")),
            str(alert.get("temperature", "")),
            str(alert.get("humidity", "")),
            str(alert.get("severity", "Warning"))
        ])

        with sqlite3.connect(DB) as con:
            already_sent = con.execute(
                "SELECT 1 FROM telegram_notifications WHERE fingerprint = ?",
                (fingerprint,)
            ).fetchone()

        if already_sent:
            continue

        message = (
            "🚨 *ENVIRONMENTAL ALERT*\n\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "📍 *LOCATION*\n"
            f"{alert.get('room_id', 'Unknown')}\n\n"
            "📡 *SENSOR DETAILS*\n"
            f"Sensor ID: {alert.get('sensor_id', 'Unknown')}\n\n"
            "🌡️ *ENVIRONMENTAL DATA*\n"
            f"Temperature: {float(alert['temperature']):.1f} °C\n"
            f"Humidity: {float(alert['humidity']):.1f}%\n\n"
            f"⚠️ *SEVERITY: {str(alert.get('severity', 'Warning')).upper()}*\n\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "🔍 *AI ANALYSIS*\n"
            "Unusual environmental conditions detected.\n\n"
            "🕒 *TIME*\n"
            f"{alert.get('timestamp', 'Unknown')}\n\n"
            "🛠️ *RECOMMENDED ACTION*\n"
            "Inspect the area and verify the sensor reading.\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "_AI Smart Monitoring System_"
        )

        success, detail = send_telegram_message(message)

        if success:
            with sqlite3.connect(DB) as con:
                con.execute(
                    """
                    INSERT OR IGNORE INTO telegram_notifications
                    (fingerprint, sent_at, message)
                    VALUES (?, ?, ?)
                    """,
                    (fingerprint, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)
                )
        else:
            print("Telegram alert failed:", detail)# 3. AI MODEL - ISOLATION FOREST
# ==========================================

def make_model():
    rng = np.random.default_rng(42)

    normal_data = np.column_stack([
        rng.normal(27, 2.2, 500),
        rng.normal(55, 8, 500)
    ])

    model = IsolationForest(
        contamination=0.08,
        random_state=42
    )

    model.fit(normal_data)

    return model


# ==========================================
# 4. ANOMALY EXPLANATION
# ==========================================

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


# ==========================================
# 5. AUTOMATIC LIVE SENSOR MONITORING
# ==========================================

def live_monitoring_panel():
    st.subheader("🔴 Live Sensor Monitoring")

    rng = np.random.default_rng()

    temperature = float(rng.normal(27, 2.2))
    humidity = float(np.clip(rng.normal(55, 8), 0, 100))

    started = time.perf_counter()
    prediction = model.predict([[temperature, humidity]])[0]
    processing_ms = (time.perf_counter() - started) * 1000
    anomaly = int(prediction == -1)
    room_id = st.session_state.get("selected_room", "Room 1")
    sensor_id = st.session_state.get("selected_sensor_id", "SENSOR-001")
    severity = classify_severity(temperature, humidity, anomaly)

    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
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

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Live Temperature", f"{temperature:.1f} °C")
    col2.metric("Live Humidity", f"{humidity:.1f}%")
    col3.metric("AI Detection", "ABNORMAL ⚠️" if anomaly else "NORMAL ✅")
    col4.metric("Edge Processing", f"{processing_ms:.3f} ms")
    st.caption(f"Monitoring location: {room_id} | Sensor: {sensor_id} | Severity: {severity}")

    if anomaly:
        st.error("Unusual environmental conditions detected!")
    else:
        st.success("Environmental conditions appear normal.")

    st.caption("Automatic simulated update every 5 seconds.")


# ==========================================
# 6. INITIALIZATION
# ==========================================

init_db()
model = make_model()


# ==========================================
# 7. MAIN DASHBOARD
# ==========================================

st.title("📡 AI-Driven Smart Monitoring System")

st.caption(
    "Simulated IoT Sensors | Local Edge-Style Processing | "
    "Machine Learning Anomaly Detection"
)


# ==========================================
# 8. SIDEBAR - SENSOR SIMULATOR
# ==========================================

with st.sidebar:
    st.header("🌡️ Sensor Simulator")

    st.write(
        "Generate readings as if they came from "
        "a temperature and humidity sensor."
    )

    temp = st.slider(
        "Temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=27.0,
        step=0.5
    )

    humidity = st.slider(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=55.0,
        step=1.0
    )

    room_id = st.selectbox(
        "Monitoring Location",
        ["Room 1", "Room 2", "Room 3", "Laboratory"]
    )
    st.session_state["selected_room"] = room_id
    sensor_map = {"Room 1": "SENSOR-001", "Room 2": "SENSOR-002", "Room 3": "SENSOR-003", "Laboratory": "SENSOR-004"}
    sensor_id = st.selectbox("Sensor ID", [sensor_map[room_id]], disabled=True,
                             help="Each monitoring location is assigned a unique simulated sensor ID.")
    st.session_state["selected_sensor_id"] = sensor_id

    st.divider()
    st.subheader("🩺 Sensor Health Simulator")
    sensor_condition = st.selectbox(
        "Simulate sensor condition",
        ["Healthy", "Disconnected", "Missing readings", "Frozen / stuck readings", "Abnormal sensor values"],
        help="Choose a fault scenario to test sensor-health monitoring. This is a software simulation."
    )
    st.session_state["sensor_condition"] = sensor_condition

    source = st.selectbox(
        "Data Source",
        [
            "Simulated IoT sensor",
            "Manual test reading"
        ]
    )

    if st.button(
        "Analyse and Save Reading",
        type="primary",
        use_container_width=True
    ):
        started = time.perf_counter()
        prediction = model.predict([[temp, humidity]])[0]
        processing_ms = (time.perf_counter() - started) * 1000
        anomaly = int(prediction == -1)
        severity = classify_severity(temp, humidity, anomaly)

        row = pd.DataFrame([{
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
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

        st.session_state["last_result"] = (
            temp,
            humidity,
            anomaly
        )

    st.divider()

    if st.button(
        "Generate 30 Sample Readings",
        use_container_width=True
    ):
        rng = np.random.default_rng()

        temperatures = rng.normal(27, 2.2, 30)
        humidities = rng.normal(55, 8, 30)

        # Add unusual test readings
        temperatures[-3:] = [43, 8, 48]
        humidities[-3:] = [92, 15, 98]

        started = time.perf_counter()
        labels = (model.predict(np.column_stack([temperatures, humidities])) == -1).astype(int)
        batch_processing_ms = (time.perf_counter() - started) * 1000

        now = datetime.now()

        rows = pd.DataFrame({
            "timestamp": [
                (
                    now - pd.Timedelta(minutes=29-i)
                ).strftime("%Y-%m-%d %H:%M:%S")
                for i in range(30)
            ],
            "temperature": np.round(
                temperatures, 1
            ),
            "humidity": np.clip(
                np.round(humidities, 1),
                0,
                100
            ),
            "anomaly": labels,
            "source": "Simulated IoT sensor",
            "room_id": ["Room 1", "Room 2", "Room 3", "Laboratory"] * 7 + ["Room 1", "Room 2"],
            "severity": [classify_severity(float(t), float(h), int(a)) for t, h, a in zip(temperatures, humidities, labels)],
            "processing_ms": round(batch_processing_ms / 30, 3),
            "sensor_id": ["SENSOR-001", "SENSOR-002", "SENSOR-003", "SENSOR-004"] * 7 + ["SENSOR-001", "SENSOR-002"]
        })

        save_rows(rows)
        notify_telegram_for_rows(rows)

        st.success(
            "30 sample readings generated successfully!"
        )

    st.divider()
    st.subheader("📡 Multi-Sensor Simulator")
    st.caption("Generate independent simulated readings for four uniquely identified sensors.")
    if st.button("Generate 40 Multi-Sensor Readings", use_container_width=True):
        sensor_rooms = {
            "SENSOR-001": "Room 1", "SENSOR-002": "Room 2",
            "SENSOR-003": "Room 3", "SENSOR-004": "Laboratory"
        }
        rng_multi = np.random.default_rng()
        multi_rows = []
        now_multi = datetime.now()
        for sid, room_name in sensor_rooms.items():
            temps = rng_multi.normal(27 + (int(sid[-1]) - 1) * 0.8, 2.0, 10)
            hums = np.clip(rng_multi.normal(55 + (int(sid[-1]) - 1) * 2, 7, 10), 0, 100)
            temps[-1] = [43, 8, 46, 12][int(sid[-1]) - 1]
            hums[-1] = [92, 15, 97, 18][int(sid[-1]) - 1]
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
        st.success("40 readings saved from 4 simulated sensors (10 readings per sensor).")

    st.divider()
    st.subheader("📨 Telegram Notifications")
    telegram_token, telegram_chat_id = get_telegram_credentials()
    if telegram_token and telegram_chat_id:
        st.success("Telegram credentials configured.")
        if st.button("Send Telegram Test Message", use_container_width=True):
            ok, detail = send_telegram_message(
               "✅ Telegram integration test successful!\n"
            )
            if ok:
                st.success(detail)
            else:
                st.error(detail)
    else:
        st.warning("Telegram is not configured yet. Add the bot token and chat ID to .streamlit/secrets.toml.")

    live_monitoring = st.toggle(
        "Enable Live Monitoring",
        value=False
    )


# ==========================================
# 9. ACTIVATE LIVE MONITORING
# ==========================================

if live_monitoring:
    st_autorefresh(
        interval=5000,
        key="live_monitoring_refresh"
    )
    live_monitoring_panel()


# Load data AFTER live monitoring saves the latest reading.
df = load_rows()


# ==========================================
# MULTI-SENSOR ANALYTICS
# ==========================================

st.divider()
st.subheader("📡 Multi-Sensor Monitoring & Comparison")
all_sensor_data = load_rows()
if not all_sensor_data.empty:
    sensor_summary = all_sensor_data.groupby(["sensor_id", "room_id"]).agg(
        Readings=("id", "count"),
        Avg_Temperature=("temperature", "mean"),
        Avg_Humidity=("humidity", "mean"),
        Anomalies=("anomaly", "sum"),
    ).reset_index()
    sensor_summary["Anomaly Rate (%)"] = (sensor_summary["Anomalies"] / sensor_summary["Readings"] * 100).round(1)
    st.dataframe(sensor_summary.round(2), use_container_width=True, hide_index=True)
    sensor_choice = st.selectbox("Inspect individual sensor", sorted(all_sensor_data["sensor_id"].dropna().unique()), key="individual_sensor_view")
    one_sensor = all_sensor_data[all_sensor_data["sensor_id"] == sensor_choice].sort_values("timestamp")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Sensor readings", len(one_sensor))
    sc2.metric("Average temperature", f"{one_sensor['temperature'].mean():.2f} °C")
    sc3.metric("Average humidity", f"{one_sensor['humidity'].mean():.2f}%")
    if len(one_sensor) > 1:
        sensor_plot = one_sensor.tail(50).copy()
        sensor_plot["timestamp"] = pd.to_datetime(sensor_plot["timestamp"], errors="coerce")
        fig_sensor = px.line(sensor_plot, x="timestamp", y=["temperature", "humidity"],
                             title=f"Recent readings — {sensor_choice}", markers=True)
        st.plotly_chart(fig_sensor, use_container_width=True)
else:
    st.info("Generate multi-sensor readings from the sidebar to view sensor-wise comparisons.")


# ==========================================
# SENSOR HEALTH AND FAILURE DETECTION
# ==========================================

st.divider()
st.subheader("🩺 Sensor Health & Failure Detection")
sensor_condition = st.session_state.get("sensor_condition", "Healthy")
selected_sensor_room = st.session_state.get("selected_room", "Room 1")
room_readings = df[df["room_id"] == selected_sensor_room] if not df.empty else pd.DataFrame()

health_status = "HEALTHY"
health_reason = "Sensor is reporting readings normally in this simulation."
health_action = "Continue monitoring."

if sensor_condition == "Disconnected":
    health_status = "DISCONNECTED"
    health_reason = "No communication is being received from the simulated sensor."
    health_action = "Check sensor power, connection and network availability."
elif sensor_condition == "Missing readings":
    health_status = "NO DATA"
    health_reason = "The expected sensor reading is missing."
    health_action = "Check the sensor and data collection pipeline."
elif sensor_condition == "Frozen / stuck readings":
    health_status = "POSSIBLE SENSOR FAULT"
    health_reason = "The simulated sensor value is assumed to be repeating without changing."
    health_action = "Inspect the sensor for a stuck value or communication issue."
elif sensor_condition == "Abnormal sensor values":
    health_status = "ABNORMAL VALUES"
    health_reason = "The simulated sensor is reporting values outside the configured safe range."
    health_action = "Verify the sensor reading and inspect the monitored environment."

health_col1, health_col2, health_col3 = st.columns(3)
health_col1.metric("Monitored Sensor", f"{selected_sensor_room} Sensor")
health_col2.metric("Sensor Health", health_status)
health_col3.metric("Stored Readings", len(room_readings))

if health_status == "HEALTHY":
    st.success(f"✅ {health_reason}")
else:
    st.error(f"🚨 SENSOR ALERT: {health_status} — {health_reason}")
st.info(f"Recommended action: {health_action}")
st.caption("Fault scenarios are simulated for demonstration; no physical IoT sensor fault is being detected.")

with st.expander("View sensor health checks"):
    checks = pd.DataFrame([
        {"Health check": "Sensor communication", "Result": "Failed" if sensor_condition == "Disconnected" else "Passed" if sensor_condition == "Healthy" else "Simulated fault"},
        {"Health check": "Reading availability", "Result": "Failed" if sensor_condition == "Missing readings" else "Passed" if sensor_condition == "Healthy" else "Simulated fault"},
        {"Health check": "Value variation", "Result": "Failed" if sensor_condition == "Frozen / stuck readings" else "Passed" if sensor_condition == "Healthy" else "Simulated fault"},
        {"Health check": "Value range", "Result": "Failed" if sensor_condition == "Abnormal sensor values" else "Passed" if sensor_condition == "Healthy" else "Simulated fault"}
    ])
    st.dataframe(checks, use_container_width=True, hide_index=True)

if not df.empty:
    st.subheader("🏢 Monitoring Location Filter")
    available_rooms = sorted(df["room_id"].dropna().unique().tolist())
    selected_rooms = st.multiselect("Select rooms to display", available_rooms, default=available_rooms)
    if selected_rooms:
        df = df[df["room_id"].isin(selected_rooms)].copy()
    else:
        st.info("Select at least one monitoring location to display readings.")
        df = df.iloc[0:0]

    if not df.empty:
        st.subheader("🔎 Advanced Filters")
        filter1, filter2 = st.columns(2)
        with filter1:
            severity_options = ["Normal", "Warning", "Critical"]
            chosen_severity = st.multiselect(
                "Filter by severity", severity_options, default=severity_options
            )
        with filter2:
            status_options = ["Normal", "Abnormal"]
            chosen_status = st.multiselect(
                "Filter by AI status", status_options, default=status_options
            )
        df = df[
            df["severity"].isin(chosen_severity)
            & df["anomaly"].map({0: "Normal", 1: "Abnormal"}).isin(chosen_status)
        ].copy()


# ==========================================
# 10. DISPLAY LATEST ANALYSIS
# ==========================================

if "last_result" in st.session_state:
    temperature, humidity, anomaly = (
        st.session_state["last_result"]
    )

    if anomaly:
        st.error(
            f"⚠️ AI detected an unusual reading: "
            f"{temperature:.1f} °C and "
            f"{humidity:.1f}% humidity."
        )
    else:
        st.success(
            f"✅ Reading appears normal: "
            f"{temperature:.1f} °C and "
            f"{humidity:.1f}% humidity."
        )


# ==========================================
# 11. DASHBOARD METRICS
# ==========================================

if df.empty:
    st.info(
        "No sensor readings available. "
        "Generate sample readings using the sidebar."
    )

else:
    total = len(df)
    anomalies = int(df["anomaly"].sum())
    normal = total - anomalies

    anomaly_percentage = (
        (anomalies / total) * 100
        if total > 0 else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Readings",
        total
    )

    col2.metric(
        "Unusual Readings",
        anomalies
    )

    col3.metric(
        "Normal Readings",
        normal
    )

    col4.metric(
        "Anomaly Rate",
        f"{anomaly_percentage:.1f}%"
    )
    col5.metric("Monitoring Rooms", df["room_id"].nunique())


    # ======================================
    # 12. LATEST SENSOR STATUS
    # ======================================

    latest = df.iloc[-1]

    st.subheader("📊 Latest Sensor Status")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Temperature",
        f"{latest['temperature']:.1f} °C"
    )

    c2.metric(
        "Humidity",
        f"{latest['humidity']:.1f}%"
    )

    c3.metric(
        "AI Status",
        "ABNORMAL ⚠️"
        if latest["anomaly"]
        else "NORMAL ✅"
    )

    st.subheader("🏢 Room-wise Monitoring")
    room_summary = df.groupby("room_id").agg(
        Readings=("id", "count"),
        Anomalies=("anomaly", "sum"),
        Avg_Temperature=("temperature", "mean"),
        Avg_Humidity=("humidity", "mean")
    ).reset_index()
    st.dataframe(room_summary.round(2), use_container_width=True, hide_index=True)

    st.subheader("⚡ Local Edge Processing Performance")
    avg_processing = float(df["processing_ms"].mean())
    max_processing = float(df["processing_ms"].max())
    perf1, perf2, perf3 = st.columns(3)
    perf1.metric("Average AI Processing", f"{avg_processing:.3f} ms")
    perf2.metric("Maximum AI Processing", f"{max_processing:.3f} ms")
    perf3.metric("Processing Location", "This computer")
    st.caption("Measured around the local Isolation Forest prediction. This is not a cloud comparison or network-latency measurement.")

    # ======================================
    # 12A. EDGE VS CLOUD-STYLE BENCHMARK
    # ======================================

    st.divider()
    st.subheader("☁️ Edge vs Cloud-Style Performance Comparison")
    st.write(
        "This experiment runs the same AI model on the same sensor readings. "
        "The edge result is measured locally. The cloud-style result adds a "
        "configurable simulated network round-trip delay; it is not a real cloud deployment."
    )

    network_delay_ms = st.slider(
        "Simulated cloud network delay per reading (ms)",
        min_value=10,
        max_value=300,
        value=80,
        step=10,
        key="cloud_simulated_delay"
    )

    benchmark_samples = df[["temperature", "humidity"]].tail(20).to_numpy(dtype=float)
    st.caption(f"Benchmark dataset: {len(benchmark_samples)} most recent readings from the current filters.")

    if st.button("Run Edge vs Cloud-Style Benchmark", type="primary"):
        edge_times = []
        cloud_times = []
        edge_predictions = []
        cloud_predictions = []

        for sample in benchmark_samples:
            # Edge: local model prediction only.
            edge_start = time.perf_counter()
            edge_prediction = model.predict([sample])[0]
            edge_elapsed = (time.perf_counter() - edge_start) * 1000
            edge_times.append(edge_elapsed)
            edge_predictions.append(int(edge_prediction))

            # Cloud-style: same model plus an explicitly simulated network delay.
            cloud_start = time.perf_counter()
            cloud_prediction = model.predict([sample])[0]
            time.sleep(network_delay_ms / 1000.0)
            cloud_elapsed = (time.perf_counter() - cloud_start) * 1000
            cloud_times.append(cloud_elapsed)
            cloud_predictions.append(int(cloud_prediction))

        edge_average = float(np.mean(edge_times))
        cloud_average = float(np.mean(cloud_times))
        latency_difference = cloud_average - edge_average
        slower_percent = (
            (latency_difference / edge_average) * 100
            if edge_average > 0 else 0.0
        )

        st.session_state["edge_cloud_benchmark"] = {
            "edge_average_ms": edge_average,
            "cloud_average_ms": cloud_average,
            "network_delay_ms": network_delay_ms,
            "sample_count": len(benchmark_samples),
            "predictions_match": edge_predictions == cloud_predictions,
            "latency_difference_ms": latency_difference,
            "slower_percent": slower_percent
        }

    benchmark_result = st.session_state.get("edge_cloud_benchmark")
    if benchmark_result:
        b1, b2, b3 = st.columns(3)
        b1.metric("Average Edge Latency", f"{benchmark_result['edge_average_ms']:.3f} ms")
        b2.metric("Average Cloud-Style Latency", f"{benchmark_result['cloud_average_ms']:.3f} ms")
        b3.metric("Added Simulated Delay", f"{benchmark_result['network_delay_ms']} ms")

        latency_chart = pd.DataFrame({
            "Processing Mode": ["Edge (local)", "Cloud-style (simulated)"],
            "Average Latency (ms)": [
                benchmark_result["edge_average_ms"],
                benchmark_result["cloud_average_ms"]
            ]
        })
        latency_fig = px.bar(
            latency_chart,
            x="Processing Mode",
            y="Average Latency (ms)",
            color="Processing Mode",
            title="Average response latency: local vs simulated cloud-style"
        )
        st.plotly_chart(latency_fig, use_container_width=True)

        st.write(
            f"**Latency difference:** {benchmark_result['latency_difference_ms']:.3f} ms "
            f"({benchmark_result['slower_percent']:.1f}% higher for the cloud-style simulation)."
        )
        st.write(
            f"**AI output consistency:** {'Both modes returned identical predictions.' if benchmark_result['predictions_match'] else 'Predictions differed; investigate the test run.'}"
        )
        st.caption(
            f"Tested {benchmark_result['sample_count']} readings. Cloud-style latency includes "
            "the artificial delay selected above and local execution of the same model. "
            "It must not be reported as measured real-cloud performance."
        )

    processing_fig = px.histogram(
        df, x="processing_ms", nbins=20,
        title="Distribution of local AI processing time",
        labels={"processing_ms": "Prediction time (milliseconds)"}
    )
    st.plotly_chart(processing_fig, use_container_width=True)


    # ======================================
    # 13. SENSOR HISTORY GRAPH
    # ======================================

    st.subheader("📈 Sensor Readings Over Time")

    long_df = df.melt(
        id_vars=["timestamp", "anomaly"],
        value_vars=[
            "temperature",
            "humidity"
        ],
        var_name="sensor",
        value_name="value"
    )

    fig = px.line(
        long_df,
        x="timestamp",
        y="value",
        color="sensor",
        markers=True,
        title="Temperature and Humidity History"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ======================================
    # 14. INDIVIDUAL SENSOR GRAPHS
    # ======================================

    st.subheader("Individual Sensor Analysis")

    graph1, graph2 = st.columns(2)

    with graph1:
        temperature_fig = px.scatter(
            df,
            x="timestamp",
            y="temperature",
            color=df["anomaly"].map({
                0: "Normal",
                1: "Abnormal"
            }),
            color_discrete_map={
                "Normal": "#2ECC71",
                "Abnormal": "#E74C3C"
            },
            title="Temperature History",
            labels={
                "color": "AI Status",
                "temperature": "Temperature (°C)"
            }
        )

        temperature_fig.update_traces(
            marker=dict(size=9)
        )

        st.plotly_chart(
            temperature_fig,
            use_container_width=True
        )

    with graph2:
        humidity_fig = px.scatter(
            df,
            x="timestamp",
            y="humidity",
            color=df["anomaly"].map({
                0: "Normal",
                1: "Abnormal"
            }),
            color_discrete_map={
                "Normal": "#2ECC71",
                "Abnormal": "#E74C3C"
            },
            title="Humidity History",
            labels={
                "color": "AI Status",
                "humidity": "Humidity (%)"
            }
        )

        humidity_fig.update_traces(
            marker=dict(size=9)
        )

        st.plotly_chart(
            humidity_fig,
            use_container_width=True
        )


    # ======================================
    # 15. AI ANOMALY RESULTS
    # ======================================

    st.subheader("🤖 AI Anomaly Detection Results")

    display = df[
        [
            "timestamp",
            "temperature",
            "humidity",
            "anomaly",
            "source",
            "room_id",
            "sensor_id",
            "severity",
            "processing_ms"
        ]
    ].copy()

    display["AI Status"] = display[
        "anomaly"
    ].map({
        0: "Normal",
        1: "Abnormal"
    })

    display["Explanation"] = display.apply(
        lambda row: (
            explain_anomaly(row)
            if row["anomaly"] == 1
            else "Within learned normal pattern"
        ),
        axis=1
    )

    display = display.drop(
        columns=["anomaly"]
    )

    st.dataframe(
        display.sort_values(
            "timestamp",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


    # ======================================
    # 16. ANOMALY ALERTS
    # ======================================

    abnormal = df[
        df["anomaly"] == 1
    ]

    if not abnormal.empty:
        st.warning(
            f"⚠️ Alert: {len(abnormal)} "
            f"unusual readings detected!"
        )
    else:
        st.success(
            "✅ No unusual readings detected."
        )

    # ======================================
    # 16A. SMART RECOMMENDATIONS
    # ======================================

    st.subheader("🧠 Smart Recommendations")
    if abnormal.empty:
        st.info("All displayed readings are classified as normal. Continue monitoring.")
    else:
        for _, alert_row in abnormal.sort_values("timestamp", ascending=False).head(5).iterrows():
            reason = explain_anomaly(alert_row)
            if alert_row["severity"] == "Critical":
                action = "Critical condition: inspect the area immediately and verify the sensor reading."
            elif "temperature" in reason.lower():
                action = "Check ventilation, cooling, heating equipment and sensor placement."
            elif "humidity" in reason.lower():
                action = "Check ventilation, moisture sources and dehumidification/humidification equipment."
            else:
                action = "Recheck the sensor and compare with nearby monitoring locations."
            st.write(f"**{alert_row['room_id']} — {alert_row['severity']}:** {reason}. {action}")

    # ======================================
    # 16B. ALERT HISTORY
    # ======================================

    st.subheader("📨 Telegram Notification History")
    with sqlite3.connect(DB) as con:
        telegram_log = pd.read_sql_query(
            "SELECT sent_at AS 'Sent At', message AS 'Message' "
            "FROM telegram_notifications ORDER BY sent_at DESC LIMIT 50",
            con
        )
    if telegram_log.empty:
        st.caption("No Telegram alerts have been sent yet.")
    else:
        st.dataframe(telegram_log, use_container_width=True, hide_index=True)

    st.subheader("🧾 Alert History")
    alert_history = df[df["anomaly"] == 1].copy()
    if alert_history.empty:
        st.caption("No anomaly events in the current filter selection.")
    else:
        alert_history["Reason"] = alert_history.apply(explain_anomaly, axis=1)
        alert_history = alert_history[["timestamp", "room_id", "temperature", "humidity", "severity", "Reason"]]
        st.dataframe(alert_history.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

    # ======================================
    # 16C. ROOM COMPARISON
    # ======================================

    st.subheader("📊 Room Anomaly Comparison")
    comparison = df.groupby("room_id").agg(
        Total_Readings=("id", "count"),
        Anomalies=("anomaly", "sum")
    ).reset_index()
    comparison["Anomaly_Rate_Percent"] = (comparison["Anomalies"] / comparison["Total_Readings"] * 100).round(1)
    comparison_fig = px.bar(
        comparison, x="room_id", y="Anomaly_Rate_Percent",
        title="Anomaly rate by monitoring location",
        labels={"room_id": "Monitoring location", "Anomaly_Rate_Percent": "Anomaly rate (%)"}
    )
    st.plotly_chart(comparison_fig, use_container_width=True)


    # ======================================
    # 16D. TEMPERATURE AND HUMIDITY FORECAST
    # ======================================

    st.divider()
    st.subheader("🔮 Temperature & Humidity Forecast")
    st.write(
        "A simple linear-trend model uses recent historical readings to estimate "
        "the next five minutes. This is an experimental forecast, not a guarantee."
    )

    forecast_room = st.selectbox(
        "Choose a room for forecasting",
        sorted(load_rows()["room_id"].dropna().unique().tolist()) if not load_rows().empty else ["Room 1"],
        key="forecast_room_select"
    )
    forecast_source = load_rows()
    forecast_source = forecast_source[forecast_source["room_id"] == forecast_room].copy()
    forecast_source["timestamp_dt"] = pd.to_datetime(forecast_source["timestamp"], errors="coerce")
    forecast_source = forecast_source.dropna(subset=["timestamp_dt"]).sort_values(["timestamp_dt", "id"]).tail(30)

    st.caption(f"Forecast data: {len(forecast_source)} recent readings from {forecast_room}.")

    if len(forecast_source) < 6:
        st.info("At least 6 saved readings are needed for a basic forecast. Generate sample readings or enable live monitoring first.")
    elif st.button("Generate 5-Minute Forecast", type="primary", key="generate_forecast"):
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
        st.session_state["environment_forecast"] = {
            "room": forecast_room,
            "history": forecast_source.copy(),
            "future": forecast_table,
            "temp_mae": temp_mae,
            "humidity_mae": humidity_mae,
            "backtest_count": len(temp_actual_test)
        }

    forecast_result = st.session_state.get("environment_forecast")
    if forecast_result and forecast_result["room"] == forecast_room:
        f1, f2 = st.columns(2)
        f1.metric("Temperature MAE (historical test)", f"{forecast_result['temp_mae']:.2f} °C" if np.isfinite(forecast_result["temp_mae"]) else "N/A")
        f2.metric("Humidity MAE (historical test)", f"{forecast_result['humidity_mae']:.2f}%" if np.isfinite(forecast_result["humidity_mae"]) else "N/A")
        st.caption(f"Walk-forward back-test points: {forecast_result['backtest_count']}. MAE is historical test error, not a guarantee of future accuracy.")
        st.dataframe(forecast_result["future"], use_container_width=True, hide_index=True)

        history = forecast_result["history"].copy()
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

        temp_chart_data = pd.DataFrame({
            "Time": list(history["timestamp_dt"]) + list(forecast_result["future"]["Forecast time"]),
            "Actual": list(temp_hist) + [None] * 5,
            "Predicted": temp_backtest + list(forecast_result["future"]["Predicted temperature (°C)"])
        })
        humidity_chart_data = pd.DataFrame({
            "Time": list(history["timestamp_dt"]) + list(forecast_result["future"]["Forecast time"]),
            "Actual": list(humidity_hist) + [None] * 5,
            "Predicted": humidity_backtest + list(forecast_result["future"]["Predicted humidity (%)"])
        })
        temp_forecast_fig = px.line(temp_chart_data, x="Time", y=["Actual", "Predicted"], markers=True, title="Temperature: actual vs predicted", labels={"value": "Temperature (°C)"})
        humidity_forecast_fig = px.line(humidity_chart_data, x="Time", y=["Actual", "Predicted"], markers=True, title="Humidity: actual vs predicted", labels={"value": "Humidity (%)"})
        st.plotly_chart(temp_forecast_fig, use_container_width=True)
        st.plotly_chart(humidity_forecast_fig, use_container_width=True)

        latest_temp_prediction = float(forecast_result["future"].iloc[0]["Predicted temperature (°C)"])
        latest_humidity_prediction = float(forecast_result["future"].iloc[0]["Predicted humidity (%)"])
        if latest_temp_prediction >= 40 or latest_temp_prediction <= 5 or latest_humidity_prediction >= 95 or latest_humidity_prediction <= 10:
            st.warning("The first forecasted point is outside the prototype's configured extreme-value thresholds. Verify conditions and sensor readings.")
        else:
            st.info("The first forecasted point is within the prototype's configured extreme-value thresholds.")
        st.caption("Forecasts use a simple linear trend fitted to recent simulated/stored readings. They are intended for demonstration and should not be used for safety-critical decisions.")

    # ======================================
    # 17. CSV DOWNLOAD
    # ======================================

    st.divider()

    st.subheader("📥 Download Sensor Reports")

    export_df = df.copy()

    export_df["AI Status"] = export_df[
        "anomaly"
    ].map({
        0: "Normal",
        1: "Abnormal"
    })

    export_df["Explanation"] = export_df.apply(
        lambda row: (
            explain_anomaly(row)
            if row["anomaly"] == 1
            else "Within learned normal pattern"
        ),
        axis=1
    )

    export_df = export_df.drop(
        columns=["anomaly"]
    )

    download1, download2 = st.columns(2)

    with download1:
        st.download_button(
            label="Download All Readings (CSV)",
            data=export_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="all_sensor_readings.csv",
            mime="text/csv",
            use_container_width=True
        )

    with download2:
        abnormal_export = export_df[
            export_df["AI Status"] == "Abnormal"
        ]

        st.download_button(
            label="Download Anomalies Only (CSV)",
            data=abnormal_export.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="anomaly_readings.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==========================================
# 18. PDF PROJECT REPORT GENERATOR
# ==========================================

st.divider()
st.subheader("📄 Generate UROP Project PDF Report")
st.write("Create a downloadable summary of sensor readings, room statistics, AI anomalies, sensor-health simulation, benchmark results and forecast results.")

if st.button("Generate PDF Project Report", type="primary"):
    current_report_data = load_rows()
    pdf_bytes = build_pdf_report(
        current_report_data,
        health_status,
        health_reason,
        health_action,
        st.session_state.get("edge_cloud_benchmark"),
        st.session_state.get("environment_forecast")
    )
    st.session_state["generated_project_pdf"] = pdf_bytes
    st.success("PDF report generated successfully. Click the download button below.")

if st.session_state.get("generated_project_pdf"):
    st.download_button(
        "⬇️ Download UROP Project Report (PDF)",
        data=st.session_state["generated_project_pdf"],
        file_name=f"AI_Smart_Monitoring_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )


# ==========================================
# 19. PROJECT ARCHITECTURE INFORMATION
# ==========================================

st.divider()

st.subheader("⚙️ How This Prototype Works")

st.markdown("""
### 1. IoT Layer
Temperature and humidity readings are simulated because
physical sensor hardware is not connected yet.

### 2. Edge Processing Layer
Sensor readings are processed locally on this computer.

### 3. Artificial Intelligence Layer
The Isolation Forest machine-learning algorithm
identifies unusual sensor readings.

### 4. Database Layer
SQLite stores sensor readings, room identifiers, severity labels and processing times.

### 5. Visualization Layer
Streamlit displays room-wise sensor values, graphs, severity alerts,
local processing performance, sensor-health fault simulation,
and downloadable CSV reports.

---

**Project Status:** Software prototype.

**Hardware Status:** Physical IoT sensor not connected.

**Processing:** Local computer-based processing.
""")

st.caption(
    "AI-Driven Smart Monitoring System | UROP Project"
)
