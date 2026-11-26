"""
AI-Driven Smart Monitoring System Using IoT & Edge Computing
Streamlit dashboard: navigation, pages and layout.

Run with:  python -m streamlit run app.py

Backend logic lives in backend.py and the design system in ui.py.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import backend as be
import ui


# ==========================================
# 1. PAGE CONFIG & BOOTSTRAP
# ==========================================

st.set_page_config(
    page_title="AI Smart Monitoring",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()
be.init_db()
model = be.get_model()

LIVE_INTERVAL_MS = 5000
TEMP_BANDS = [(0, 5, ui.RED), (5, 15, ui.AMBER), (15, 35, ui.GREEN), (35, 40, ui.AMBER), (40, 60, ui.RED)]
HUM_BANDS = [(0, 10, ui.RED), (10, 25, ui.AMBER), (25, 85, ui.GREEN), (85, 95, ui.AMBER), (95, 100, ui.RED)]


# ==========================================
# 2. ACTIONS (button callbacks run before the page re-renders,
#    so every page sees the newly saved readings immediately)
# ==========================================

def _toast(msg, icon_name="✅"):
    st.session_state.setdefault("_toasts", []).append((msg, icon_name))


def action_generate_samples():
    n = be.generate_sample_readings(model)
    _toast(f"{n} sample readings generated (including 3 unusual test readings).")


def action_generate_multi():
    n = be.generate_multi_sensor_readings(model)
    _toast(f"{n} readings saved from 4 simulated sensors (10 per sensor).")


def action_manual_reading():
    room = st.session_state.get("sim_room", "Room 1")
    result = be.analyse_manual_reading(
        model,
        float(st.session_state.get("sim_temp", 27.0)),
        float(st.session_state.get("sim_hum", 55.0)),
        st.session_state.get("sim_source", be.DATA_SOURCES[0]),
        room,
        be.SENSOR_MAP[room],
    )
    st.session_state["last_result"] = result
    _toast(f"Reading analysed: {result['severity']}", "⚠️" if result["anomaly"] else "✅")


# ==========================================
# 3. SHARED HELPERS
# ==========================================

def filtered(df):
    """Apply the global location / severity / AI-status filters."""
    if df.empty:
        return df
    rooms = st.session_state.get("f_rooms") or []
    sev = st.session_state.get("f_sev") or []
    status = st.session_state.get("f_status") or []
    out = df
    if rooms:
        out = out[out["room_id"].isin(rooms)]
    if sev:
        out = out[out["severity"].isin(sev)]
    if status:
        out = out[out["anomaly"].map({0: "Normal", 1: "Abnormal"}).isin(status)]
    return out.copy()


def header(title, subtitle, eyebrow, filters=False, right=None):
    """Page title with an optional Filters popover or custom right-side control."""
    if not filters and right is None:
        ui.page_title(title, subtitle, eyebrow)
        return
    c1, c2 = st.columns([5, 1.4], vertical_alignment="bottom")
    with c1:
        ui.page_title(title, subtitle, eyebrow)
    with c2:
        if filters:
            active = sum(bool(st.session_state.get(k)) for k in ("f_rooms", "f_sev", "f_status"))
            label = f"Filters · {active}" if active else "Filters"
            with st.popover(label, icon=":material/tune:", width="stretch"):
                rooms = sorted(set(be.ROOMS) | set(ALL["room_id"].dropna().unique()) if not ALL.empty else be.ROOMS)
                st.multiselect("Locations", rooms, default=[], key="f_rooms",
                               placeholder="All locations", persist_state="session")
                st.multiselect("Severity", be.SEVERITIES, default=[], key="f_sev",
                               placeholder="All severities", persist_state="session")
                st.multiselect("AI status", ["Normal", "Abnormal"], default=[], key="f_status",
                               placeholder="Normal and abnormal", persist_state="session")
                st.caption("Filters apply to Overview, AI Anomaly Detection, Benchmark, Alert History and Reports.")
        else:
            right()


def sensor_snapshot(df):
    """Latest values and counts for each of the four virtual sensors."""
    snap = []
    for room in be.ROOMS:
        sid = be.SENSOR_MAP[room]
        part = df[df["sensor_id"] == sid] if not df.empty else df
        if part.empty:
            snap.append(dict(room=room, sensor_id=sid, temperature=None, humidity=None,
                             severity=None, anomalies=0, readings=0, last_ts=None))
            continue
        last = part.iloc[-1]
        snap.append(dict(room=room, sensor_id=sid, temperature=float(last["temperature"]),
                         humidity=float(last["humidity"]), severity=str(last["severity"]),
                         anomalies=int(part["anomaly"].sum()), readings=len(part),
                         last_ts=str(last["timestamp"])))
    return snap


def sensor_cards(df):
    ui.kpi_grid([ui.sensor_card(**s) for s in sensor_snapshot(df)], kind="sensors")


def no_data(body="Use Quick actions in the sidebar to generate sample readings, or turn on live monitoring."):
    ui.empty_state("No sensor readings yet", body)


NUM = st.column_config.NumberColumn
TEXT = st.column_config.TextColumn


# ==========================================
# 4. PAGES
# ==========================================

def page_overview():
    header("Overview", "Real-time health of every monitored location, powered by on-device anomaly detection.",
           "Dashboard", filters=True)
    df = filtered(ALL)
    if df.empty:
        no_data()
        return

    total = len(df)
    anomalies = int(df["anomaly"].sum())
    normal = total - anomalies
    critical = int((df["severity"] == "Critical").sum())
    rate = anomalies / total * 100 if total else 0
    rooms = df["room_id"].nunique()
    ui.kpi_grid([
        ui.kpi_card("Total readings", f"{total:,}", f"{df['source'].nunique()} data sources", "database", "accent"),
        ui.kpi_card("Normal readings", f"{normal:,}", f"{normal / total * 100:.1f}% of all readings", "check", "normal"),
        ui.kpi_card("Anomalies detected", f"{anomalies:,}", f"Anomaly rate {rate:.1f}%", "alert", "warning"),
        ui.kpi_card("Critical alerts", f"{critical:,}", "Needs immediate attention" if critical else "None in this selection",
                    "critical", "critical"),
        ui.kpi_card("Monitoring locations", rooms, f"{df['sensor_id'].nunique()} sensors reporting", "pin", "teal"),
    ])

    c1, c2 = st.columns([2.2, 1], gap="medium")
    with c1:
        ui.show_chart(ui.dual_history_chart(df.tail(300)), key="ov_hist")
    with c2:
        ui.show_chart(ui.severity_donut(df, height=360), key="ov_donut")

    c3, c4 = st.columns([1, 1.25], gap="medium")
    with c3:
        comp = df.groupby("room_id").agg(Total=("id", "count"), Anomalies=("anomaly", "sum")).reset_index()
        comp["Rate"] = (comp["Anomalies"] / comp["Total"] * 100).round(1)
        colors = [ui.RED if r >= 20 else ui.AMBER if r >= 8 else ui.GREEN for r in comp["Rate"]]
        ui.show_chart(ui.bar_chart(comp["room_id"], comp["Rate"], "Anomaly rate by location", "%", colors, 340),
                      key="ov_rooms")
    with c4:
        ui.section("Recent alerts", "Latest AI-detected anomalies")
        recent = df[df["anomaly"] == 1].sort_values(["timestamp", "id"], ascending=False).head(4)
        if recent.empty:
            ui.callout("No anomalies in the current selection. All readings match the learned normal pattern.",
                       "normal", "check")
        else:
            ui.render("".join(
                ui.alert_row(f"{r['room_id']} · {r['sensor_id']}",
                             f"{r['temperature']:.1f} °C, {r['humidity']:.1f}% · {be.explain_anomaly(r)}",
                             str(r["timestamp"])[11:19], str(r["timestamp"])[:10], r["severity"])
                for _, r in recent.iterrows()))


def page_live():
    def room_picker():
        st.selectbox("Live sensor location", be.ROOMS, key="sim_room", persist_state="session",
                     help="Automatic live readings and manual readings are saved for this location.")

    header("Live Monitoring", "Streaming temperature and humidity with instant AI classification at the edge.",
           "Monitor", right=room_picker)
    room = st.session_state.get("sim_room", "Room 1")
    sensor_id = be.SENSOR_MAP[room]

    if LIVE:
        current, label = LIVE, "Live reading"
        ui.callout(f"Live monitoring is on. A new simulated reading from {sensor_id} is analysed and saved "
                   f"every {LIVE_INTERVAL_MS // 1000} seconds.", "normal", "radio")
    else:
        room_rows = ALL[ALL["room_id"] == room]
        if room_rows.empty:
            current = None
        else:
            r = room_rows.iloc[-1]
            current = dict(temperature=float(r["temperature"]), humidity=float(r["humidity"]),
                           anomaly=int(r["anomaly"]), severity=r["severity"],
                           processing_ms=float(r["processing_ms"]), room_id=r["room_id"],
                           sensor_id=r["sensor_id"], timestamp=r["timestamp"], source=r["source"])
        label = "Latest stored reading"
        ui.callout("Live monitoring is paused. Turn it on from the sidebar. Showing the most recent stored reading.",
                   "neutral", "clock")

    if current is None:
        no_data(f"No readings stored for {room} yet. Turn on live monitoring or analyse a manual reading below.")
    else:
        bar = ui.SEVERITY_COLOR.get(current["severity"], ui.ACCENT)
        g1, g2, g3 = st.columns([1, 1, 1.05], gap="medium", vertical_alignment="center")
        with g1:
            ui.show_chart(ui.gauge(current["temperature"], "Temperature", 60, TEMP_BANDS, " °C", bar, 268), key="g_t")
        with g2:
            ui.show_chart(ui.gauge(current["humidity"], "Humidity", 100, HUM_BANDS, " %", bar, 268), key="g_h")
        with g3:
            status = ui.badge("Abnormal", "critical" if current["severity"] == "Critical" else "warning") \
                if current["anomaly"] else ui.badge("Normal", "normal")
            ui.render(ui.kv_panel([
                ("Source", ui.esc(label)),
                ("AI detection", status),
                ("Severity", ui.severity_badge(current["severity"])),
                ("Edge processing", f"{current['processing_ms']:.3f} ms"),
                ("Location", f"{ui.esc(current['room_id'])} · <span style='font-family:var(--mono)'>{ui.esc(current['sensor_id'])}</span>"),
                ("Timestamp", f"<span style='font-family:var(--mono)'>{ui.esc(current['timestamp'])}</span>"),
            ]))

    ui.section("Sensor status", "Latest reading from each virtual sensor")
    sensor_cards(ALL)

    ui.section("Reading history", "Last 60 readings, with anomalies marked by severity")
    seg = st.segmented_control("History scope", ["All locations"] + be.ROOMS, default="All locations",
                               key="live_scope", persist_state="session", label_visibility="collapsed")
    hist = ALL if seg in (None, "All locations") else ALL[ALL["room_id"] == seg]
    if hist.empty:
        no_data("No readings for this location yet.")
    else:
        hist = hist.tail(60)
        h1, h2 = st.columns(2, gap="medium")
        with h1:
            ui.show_chart(ui.history_chart(hist, "temperature", ui.ACCENT, "Temperature", "°C", TEMP_BANDS[1:4]), key="lh_t")
        with h2:
            ui.show_chart(ui.history_chart(hist, "humidity", ui.TEAL, "Humidity", "%", HUM_BANDS[1:4]), key="lh_h")

    ui.section("Manual sensor reading", f"Send a test reading from {sensor_id} through the AI model")
    with st.container(border=True):
        m1, m2, m3 = st.columns([1.3, 1.3, 1], gap="large", vertical_alignment="bottom")
        with m1:
            st.slider("Temperature (°C)", 0.0, 60.0, 27.0, 0.5, key="sim_temp", persist_state="session")
        with m2:
            st.slider("Humidity (%)", 0.0, 100.0, 55.0, 1.0, key="sim_hum", persist_state="session")
        with m3:
            st.selectbox("Data source", be.DATA_SOURCES, key="sim_source", persist_state="session")
        st.button("Analyse and save reading", type="primary", icon=":material/bolt:",
                  on_click=action_manual_reading, width="stretch")

    res = st.session_state.get("last_result")
    if res:
        tone = "normal" if not res["anomaly"] else ("critical" if res["severity"] == "Critical" else "warning")
        msg = (f"Last manual analysis · {res['temperature']:.1f} °C, {res['humidity']:.1f}% at {res['room_id']}: "
               + ("AI detected an unusual reading" if res["anomaly"] else "reading appears normal")
               + f" ({res['severity']}, {res['processing_ms']:.3f} ms).")
        ui.callout(msg, tone, "alert" if res["anomaly"] else "check")


def page_multisensor():
    def gen_button():
        st.button("Generate 40 readings", icon=":material/sensors:", on_click=action_generate_multi,
                  width="stretch", help="10 independent readings for each of the four sensors.")

    header("Multi-Sensor Monitoring", "Four uniquely identified virtual sensors, compared side by side.",
           "Monitor", right=gen_button)
    if ALL.empty:
        sensor_cards(ALL)
        no_data("Click Generate 40 readings to simulate all four sensors.")
        return

    sensor_cards(ALL)

    ui.section("Sensor comparison", "Aggregated over all stored readings")
    summary = ALL.groupby(["sensor_id", "room_id"]).agg(
        Readings=("id", "count"), Avg_Temperature=("temperature", "mean"),
        Avg_Humidity=("humidity", "mean"), Anomalies=("anomaly", "sum"),
    ).reset_index()
    summary["Anomaly Rate (%)"] = (summary["Anomalies"] / summary["Readings"] * 100).round(1)
    st.dataframe(summary.round(2), hide_index=True, column_config={
        "sensor_id": TEXT("Sensor ID"), "room_id": TEXT("Location"),
        "Avg_Temperature": NUM("Avg temperature", format="%.2f °C"),
        "Avg_Humidity": NUM("Avg humidity", format="%.2f %%"),
        "Anomaly Rate (%)": st.column_config.ProgressColumn("Anomaly rate", format="%.1f%%", min_value=0, max_value=100),
    })

    ui.section("Sensor inspector", "Drill into a single sensor")
    c1, c2 = st.columns([1, 3], gap="medium")
    with c1:
        sensors = sorted(ALL["sensor_id"].dropna().unique())
        choice = st.selectbox("Sensor", sensors, key="inspect_sensor", persist_state="session")
        one = ALL[ALL["sensor_id"] == choice]
        ui.render(ui.kv_panel([
            ("Location", ui.esc(one["room_id"].iloc[-1]) if not one.empty else "—"),
            ("Readings", f"{len(one):,}"),
            ("Avg temperature", f"{one['temperature'].mean():.2f} °C"),
            ("Avg humidity", f"{one['humidity'].mean():.2f} %"),
            ("Anomalies", f"{int(one['anomaly'].sum())}"),
        ]))
    with c2:
        if len(one) > 1:
            fig = ui.dual_history_chart(one.tail(50), height=340)
            fig.update_layout(title=f"Recent readings · {choice}")
            ui.show_chart(fig, key="ms_inspect")
        else:
            ui.empty_state("Not enough readings", "At least two readings are needed to draw a trend.", "trend")


def page_health():
    header("Sensor Health", "Simulate hardware fault scenarios and see how the monitoring pipeline responds.",
           "Monitor")
    room = st.session_state.get("sim_room", "Room 1")
    c1, c2 = st.columns([1, 2.2], gap="large")
    with c1:
        st.selectbox("Simulate sensor condition", be.SENSOR_CONDITIONS, key="sensor_condition",
                     persist_state="session",
                     help="Choose a fault scenario to test sensor-health monitoring. This is a software simulation.")
        st.caption(f"Applies to the {room} sensor ({be.SENSOR_MAP[room]}). Change the location on the Live Monitoring page.")
    condition = st.session_state.get("sensor_condition", "Healthy")
    health = be.sensor_health(condition)
    healthy = health["status"] == "HEALTHY"
    room_rows = ALL[ALL["room_id"] == room] if not ALL.empty else ALL
    with c2:
        ui.kpi_grid([
            ui.kpi_card("Monitored sensor", room, be.SENSOR_MAP[room], "radio", "accent"),
            ui.kpi_card("Sensor health", health["status"].title(), condition, "check" if healthy else "critical",
                        "normal" if healthy else "critical"),
            ui.kpi_card("Stored readings", f"{len(room_rows):,}", f"For {room}", "database", "teal"),
        ], kind="three")

    if healthy:
        ui.callout(health["reason"], "normal", "check")
    else:
        ui.callout(f"SENSOR ALERT · {health['status']}: {health['reason']}", "critical", "critical")
    ui.callout(f"Recommended action: {health['action']}", "accent", "sparkles")

    ui.section("Health checks", "Automated diagnostics for the selected sensor")
    checks = be.health_checks(condition)
    tone = {"Passed": "normal", "Failed": "critical", "Simulated fault": "warning"}
    ui.render('<div class="card">' + "".join(
        f'<div class="kv"><span>{ui.esc(r["Health check"])}</span><b>{ui.badge(r["Result"], tone[r["Result"]])}</b></div>'
        for _, r in checks.iterrows()) + "</div>")
    st.caption("Fault scenarios are simulated for demonstration; no physical IoT sensor fault is being detected.")


def page_anomaly():
    header("AI Anomaly Detection", "An Isolation Forest flags readings that fall outside the learned normal pattern. "
           "Rule-based severity is then applied on top of each detection.", "Intelligence", filters=True)
    df = filtered(ALL)
    if df.empty:
        no_data()
        return
    total = len(df)
    anomalies = int(df["anomaly"].sum())
    critical = int((df["severity"] == "Critical").sum())
    warning = int((df["severity"] == "Warning").sum())
    ui.kpi_grid([
        ui.kpi_card("Isolation Forest", "Active", f"{be.MODEL_TRAINING_SAMPLES} training samples", "sparkles", "accent"),
        ui.kpi_card("Anomalies", f"{anomalies:,}", f"{anomalies / total * 100:.1f}% of {total:,} readings", "alert", "warning"),
        ui.kpi_card("Normal", f"{total - anomalies:,}", "Within learned pattern", "check", "normal"),
        ui.kpi_card("Warning", f"{warning:,}", "Anomalous, below critical limits", "alert", "warning"),
        ui.kpi_card("Critical", f"{critical:,}", "Beyond critical thresholds", "critical", "critical"),
    ])

    c1, c2 = st.columns([1.7, 1], gap="medium")
    with c1:
        ui.show_chart(ui.scatter_temp_humidity(df, height=400), key="an_scatter")
    with c2:
        ui.show_chart(ui.severity_donut(df, height=400), key="an_donut")

    c3, c4 = st.columns([1.7, 1], gap="medium")
    with c3:
        import plotly.graph_objects as go
        by_room = df.groupby("room_id")["anomaly"].agg(["count", "sum"]).reset_index()
        fig = go.Figure([
            go.Bar(name="Normal", x=by_room["room_id"], y=by_room["count"] - by_room["sum"],
                   marker=dict(color=ui.rgba(ui.GREEN, 0.75), cornerradius=4)),
            go.Bar(name="Anomalous", x=by_room["room_id"], y=by_room["sum"],
                   marker=dict(color=ui.RED, cornerradius=4)),
        ])
        fig.update_layout(barmode="stack", title="Normal vs anomalous readings by location", height=330, bargap=0.45)
        ui.show_chart(fig, key="an_stack")
    with c4:
        ui.section("Model details")
        ui.render(ui.kv_panel([
            ("Algorithm", "Isolation Forest"),
            ("Features", "Temperature, humidity"),
            ("Contamination", f"{be.MODEL_CONTAMINATION:.0%}"),
            ("Trees", f"{model.n_estimators}"),
            ("Training data", "Simulated normal (27 °C, 55 %)"),
            ("Runs on", "This computer (edge)"),
        ]))

    ui.section("AI explanations", "Why the most recent anomalies were flagged, and what to do")
    recent = df[df["anomaly"] == 1].sort_values(["timestamp", "id"], ascending=False).head(5)
    if recent.empty:
        ui.callout("All displayed readings are classified as normal. Continue monitoring.", "normal", "check")
    else:
        ui.render("".join(
            ui.alert_row(f"{r['room_id']} · {be.explain_anomaly(r)}",
                         f"{r['temperature']:.1f} °C, {r['humidity']:.1f}% · {be.recommend_action(r)}",
                         r["sensor_id"], str(r["timestamp"])[5:19], r["severity"])
            for _, r in recent.iterrows()))

    ui.section("Recent anomaly records")
    records = be.add_explanations(df[df["anomaly"] == 1]).sort_values("timestamp", ascending=False)
    cols = ["timestamp", "room_id", "sensor_id", "temperature", "humidity", "severity", "Explanation", "processing_ms"]
    st.dataframe(records[cols], hide_index=True, height=320, column_config={
        "timestamp": TEXT("Timestamp"), "room_id": TEXT("Location"), "sensor_id": TEXT("Sensor"),
        "temperature": NUM("Temp", format="%.1f °C"), "humidity": NUM("Humidity", format="%.1f %%"),
        "severity": TEXT("Severity"), "processing_ms": NUM("AI time", format="%.3f ms"),
    })
    with st.expander("All AI detection results"):
        full = be.add_explanations(df).sort_values("timestamp", ascending=False)
        st.dataframe(full[["timestamp", "temperature", "humidity", "source", "room_id", "sensor_id", "severity",
                           "processing_ms", "AI Status", "Explanation"]], hide_index=True)


def page_benchmark():
    header("Edge vs Cloud Benchmark", "The same AI model on the same readings, run locally and with a "
           "simulated network round-trip.", "Intelligence", filters=True)
    df = filtered(ALL)
    if df.empty:
        no_data()
        return
    ui.callout("Cloud-style latency is simulated. It adds an artificial network delay to a local model run "
               "and is not a measurement of a real cloud deployment.", "warning", "cloud")

    samples = df[["temperature", "humidity"]].tail(20).to_numpy(dtype=float)
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom", gap="large")
    with c1:
        delay = st.slider("Simulated cloud network delay per reading (ms)", 10, 300, 80, 10,
                          key="cloud_simulated_delay", persist_state="session")
    with c2:
        run = st.button("Run benchmark", type="primary", icon=":material/speed:", width="stretch")
    st.caption(f"Benchmark dataset: {len(samples)} most recent readings from the current filters.")
    if run:
        with st.spinner(f"Running {len(samples)} edge and cloud-style predictions…"):
            st.session_state["edge_cloud_benchmark"] = be.run_benchmark(model, samples, delay)

    b = st.session_state.get("edge_cloud_benchmark")
    if b:
        speedup = b["cloud_average_ms"] / b["edge_average_ms"] if b["edge_average_ms"] > 0 else 0
        ui.kpi_grid([
            ui.kpi_card("Average edge latency", f"{b['edge_average_ms']:.3f}", "Local Isolation Forest", "cpu", "normal", "ms"),
            ui.kpi_card("Average cloud-style latency", f"{b['cloud_average_ms']:.1f}", "Simulated, includes delay", "cloud", "accent", "ms"),
            ui.kpi_card("Simulated network delay", f"{b['network_delay_ms']}", "Per reading", "radio", "neutral", "ms"),
            ui.kpi_card("Edge advantage", f"{speedup:,.0f}×", f"{b['latency_difference_ms']:.1f} ms saved per reading", "zap", "warning"),
        ])
        g1, g2 = st.columns([1, 1.4], gap="medium")
        with g1:
            ui.show_chart(ui.bar_chart(["Edge (local)", "Cloud-style (simulated)"],
                                       [b["edge_average_ms"], b["cloud_average_ms"]],
                                       "Average response latency", " ms", [ui.GREEN, ui.ACCENT], 340, "{:.2f}"),
                          key="bm_bar")
        with g2:
            if b.get("edge_times"):
                import plotly.graph_objects as go
                idx = list(range(1, len(b["edge_times"]) + 1))
                fig = go.Figure([
                    go.Scatter(x=idx, y=b["cloud_times"], name="Cloud-style (simulated)", mode="lines+markers",
                               line=dict(color=ui.ACCENT, width=2), marker=dict(size=5)),
                    go.Scatter(x=idx, y=b["edge_times"], name="Edge (local)", mode="lines+markers",
                               line=dict(color=ui.GREEN, width=2), marker=dict(size=5)),
                ])
                fig.update_layout(title="Latency per reading (log scale)", height=340)
                fig.update_yaxes(type="log", ticksuffix=" ms", dtick=1)
                fig.update_xaxes(title_text="Reading #")
                fig.update_layout(margin=dict(b=48))
                ui.show_chart(fig, key="bm_line")
        same = b["predictions_match"]
        ui.callout(("Both modes returned identical predictions, so moving inference to the edge did not change "
                    "the AI output." if same else "Predictions differed between modes. Investigate the test run.")
                   + f" Tested {b['sample_count']} readings; cloud-style was {b['slower_percent']:.0f}% slower.",
                   "normal" if same else "critical", "check" if same else "alert")
    else:
        ui.empty_state("No benchmark run yet", "Pick a simulated delay and click Run benchmark.", "zap")

    ui.section("Local edge processing performance", "Measured around every stored Isolation Forest prediction")
    avg_p, max_p = float(df["processing_ms"].mean()), float(df["processing_ms"].max())
    p1, p2 = st.columns([1, 2], gap="medium")
    with p1:
        ui.render(ui.kv_panel([
            ("Average AI processing", f"{avg_p:.3f} ms"),
            ("Maximum AI processing", f"{max_p:.3f} ms"),
            ("Processing location", "This computer"),
            ("Readings measured", f"{len(df):,}"),
        ]))
        st.caption("Measured locally. This is not a cloud comparison or a network-latency measurement.")
    with p2:
        import plotly.graph_objects as go
        fig = go.Figure(go.Histogram(x=df["processing_ms"], nbinsx=20,
                                     marker=dict(color=ui.rgba(ui.ACCENT, 0.8), line=dict(color=ui.BG, width=1))))
        fig.update_layout(title="Distribution of local AI processing time", height=300, hovermode="closest", bargap=0.05, margin=dict(t=52, b=48))
        fig.update_xaxes(title_text="Prediction time (ms)")
        ui.show_chart(fig, key="bm_hist")


def page_forecast():
    header("Temperature & Humidity Forecast", "A linear trend fitted to recent readings estimates the next five "
           "minutes, with walk-forward back-testing to measure accuracy.", "Intelligence")
    if ALL.empty:
        no_data()
        return
    rooms = sorted(ALL["room_id"].dropna().unique().tolist())
    c1, c2, c3 = st.columns([1.3, 1, 2], vertical_alignment="bottom", gap="medium")
    with c1:
        room = st.selectbox("Location", rooms, key="forecast_room_select", persist_state="session")
    source = be.prepare_forecast_source(ALL, room)
    enough = len(source) >= 6
    with c2:
        go_btn = st.button("Generate forecast", type="primary", icon=":material/insights:",
                           disabled=not enough, width="stretch")
    with c3:
        st.caption(f"Forecast data: {len(source)} recent readings from {room}.")
    if not enough:
        ui.callout("At least 6 saved readings are needed for a basic forecast. Generate sample readings or "
                   "turn on live monitoring first.", "warning", "alert")
    if go_btn and enough:
        st.session_state["environment_forecast"] = be.compute_forecast(source, room)

    fr = st.session_state.get("environment_forecast")
    if not fr or fr["room"] != room:
        if enough:
            ui.empty_state("No forecast yet", f"Click Generate forecast to predict the next 5 minutes for {room}.", "trend")
        return

    ui.kpi_grid([
        ui.kpi_card("Temperature MAE", f"{fr['temp_mae']:.2f}" if np.isfinite(fr["temp_mae"]) else "N/A",
                    "Historical back-test error", "thermometer", "accent", "°C"),
        ui.kpi_card("Humidity MAE", f"{fr['humidity_mae']:.2f}" if np.isfinite(fr["humidity_mae"]) else "N/A",
                    "Historical back-test error", "droplet", "teal", "%"),
        ui.kpi_card("Back-test points", fr["backtest_count"], "Walk-forward evaluation", "trend", "neutral"),
        ui.kpi_card("Next-minute temp", f"{fr['future'].iloc[0]['Predicted temperature (°C)']:.1f}",
                    str(fr["future"].iloc[0]["Forecast time"])[11:16], "clock", "warning", "°C"),
    ])

    history = fr["history"]
    t_hist, h_hist, t_bt, h_bt = be.forecast_backtest_series(history)
    times = list(history["timestamp_dt"])
    fut = fr["future"]
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        ui.show_chart(ui.forecast_chart(times, list(t_hist), t_bt, list(fut["Forecast time"]),
                                        list(fut["Predicted temperature (°C)"]), "Temperature · actual vs predicted",
                                        "°C", ui.ACCENT), key="fc_t")
    with g2:
        ui.show_chart(ui.forecast_chart(times, list(h_hist), h_bt, list(fut["Forecast time"]),
                                        list(fut["Predicted humidity (%)"]), "Humidity · actual vs predicted",
                                        "%", ui.TEAL), key="fc_h")

    t1, t2 = st.columns([1.3, 1], gap="medium")
    with t1:
        ui.section("Five-minute prediction")
        st.dataframe(fut, hide_index=True, column_config={
            "Forecast time": st.column_config.DatetimeColumn("Forecast time", format="HH:mm:ss"),
            "Predicted temperature (°C)": NUM(format="%.2f °C"),
            "Predicted humidity (%)": NUM(format="%.2f %%"),
        })
    with t2:
        ui.section("Threshold check")
        lt = float(fut.iloc[0]["Predicted temperature (°C)"])
        lh = float(fut.iloc[0]["Predicted humidity (%)"])
        if lt >= 40 or lt <= 5 or lh >= 95 or lh <= 10:
            ui.callout("The first forecasted point is outside the prototype's configured extreme-value thresholds. "
                       "Verify conditions and sensor readings.", "warning", "alert")
        else:
            ui.callout("The first forecasted point is within the prototype's configured extreme-value thresholds.",
                       "normal", "check")
        ui.callout("Limitations: a simple linear trend fitted to recent simulated or stored readings. MAE is "
                   "historical test error, not a guarantee of future accuracy. Not for safety-critical decisions.",
                   "neutral", "layers")


def page_alerts():
    header("Alert History", "Every AI-detected anomaly, with its Telegram delivery status.", "Operations", filters=True)
    df = filtered(ALL)
    alerts = df[df["anomaly"] == 1].copy() if not df.empty else df
    if alerts.empty:
        ui.empty_state("No alerts", "No anomaly events in the current selection.", "bell")
        return
    alerts["Reason"] = alerts.apply(be.explain_anomaly, axis=1)
    alerts = be.attach_delivery_status(alerts).sort_values(["timestamp", "id"], ascending=False)
    tg_on = be.telegram_configured()
    if not tg_on:
        alerts.loc[alerts["Delivery"] == "Not sent", "Delivery"] = "Not configured"
    alerts["Delivered at"] = alerts["Delivered at"].fillna("—")
    delivered = int((alerts["Delivery"] == "Delivered").sum())
    ui.kpi_grid([
        ui.kpi_card("Total alerts", f"{len(alerts):,}", "In current selection", "bell", "accent"),
        ui.kpi_card("Warning", f"{int((alerts['severity'] == 'Warning').sum()):,}", "Anomalous, below critical limits",
                    "alert", "warning"),
        ui.kpi_card("Critical", f"{int((alerts['severity'] == 'Critical').sum()):,}", "Beyond critical thresholds",
                    "critical", "critical"),
        ui.kpi_card("Delivered to Telegram", f"{delivered:,}",
                    "Bot configured" if tg_on else "Telegram not configured", "send", "normal" if tg_on else "neutral"),
    ])

    view = st.segmented_control("Show", ["All", "Warning", "Critical"], default="All", key="alert_view",
                                persist_state="session", label_visibility="collapsed")
    shown = alerts if view in (None, "All") else alerts[alerts["severity"] == view]
    ui.section("Latest alerts", f"{len(shown):,} alerts · showing the 6 most recent")
    ui.render("".join(
        ui.alert_row(f"{r['room_id']} · {r['sensor_id']}",
                     f"{r['temperature']:.1f} °C, {r['humidity']:.1f}% · {r['Reason']}",
                     str(r["timestamp"]), "✓ Telegram delivered" if r["Delivery"] == "Delivered" else f"Telegram: {r['Delivery'].lower()}",
                     r["severity"])
        for _, r in shown.head(6).iterrows()))

    ui.section("All alert records")
    st.dataframe(shown[["timestamp", "severity", "room_id", "sensor_id", "temperature", "humidity", "Reason",
                        "Delivery", "Delivered at"]], hide_index=True, height=340, column_config={
        "timestamp": TEXT("Timestamp"), "severity": TEXT("Severity"), "room_id": TEXT("Location"),
        "sensor_id": TEXT("Sensor ID"), "temperature": NUM("Temp", format="%.1f °C"),
        "humidity": NUM("Humidity", format="%.1f %%"),
    })

    ui.section("Telegram notification log", "Messages actually sent by the bot (latest 50)")
    log = be.load_telegram_log(50)
    if log.empty:
        st.caption("No Telegram alerts have been sent yet." if tg_on else
                   "Telegram is not configured. See Settings to connect a bot.")
    else:
        st.dataframe(log[["sent_at", "message"]].rename(columns={"sent_at": "Sent at", "message": "Message"}),
                     hide_index=True)


def page_reports():
    header("Reports & Exports", "Download sensor data and generate the UROP project PDF report.", "Operations",
           filters=True)
    df = filtered(ALL)
    if not df.empty:
        ui.kpi_grid([
            ui.kpi_card("Readings in selection", f"{len(df):,}", "Exported to CSV", "database", "accent"),
            ui.kpi_card("Anomaly records", f"{int(df['anomaly'].sum()):,}", "Included in anomaly export", "alert", "warning"),
            ui.kpi_card("Date range", str(df["timestamp"].min())[:10],
                        f"to {str(df['timestamp'].max())[:16]}", "clock", "neutral"),
        ], kind="three")

    c1, c2, c3 = st.columns(3, gap="medium")
    export = None
    if not df.empty:
        export = be.add_explanations(df).drop(columns=["anomaly", "ts"], errors="ignore")
    with c1:
        with st.container(border=True):
            ui.render(f'<div class="kpi-top"><b>All readings</b>{ui.icon("file", 18, ui.ACCENT)}</div>'
                      '<div class="kpi-sub">Every reading in the current filter selection, with AI status and explanation.</div>')
            st.download_button("Download CSV", data=export.to_csv(index=False).encode("utf-8") if export is not None else b"",
                               file_name="all_sensor_readings.csv", mime="text/csv", icon=":material/download:",
                               width="stretch", disabled=export is None)
    with c2:
        with st.container(border=True):
            ui.render(f'<div class="kpi-top"><b>Anomalies only</b>{ui.icon("alert", 18, ui.AMBER)}</div>'
                      '<div class="kpi-sub">Readings the AI classified as abnormal, ready for incident review.</div>')
            ab = export[export["AI Status"] == "Abnormal"] if export is not None else None
            st.download_button("Download CSV", data=ab.to_csv(index=False).encode("utf-8") if ab is not None else b"",
                               file_name="anomaly_readings.csv", mime="text/csv", icon=":material/download:",
                               width="stretch", disabled=ab is None, key="dl_anom")
    with c3:
        with st.container(border=True):
            ui.render(f'<div class="kpi-top"><b>UROP project report</b>{ui.icon("file", 18, ui.TEAL)}</div>'
                      '<div class="kpi-sub">PDF with readings, room statistics, sensor health, benchmark and forecast.</div>')
            if st.button("Generate PDF report", type="primary", icon=":material/picture_as_pdf:", width="stretch"):
                with st.spinner("Building PDF…"):
                    st.session_state["generated_project_pdf"] = be.build_pdf_report(
                        be.load_rows(), HEALTH["status"], HEALTH["reason"], HEALTH["action"],
                        st.session_state.get("edge_cloud_benchmark"), st.session_state.get("environment_forecast"))
                _toast("PDF report generated. Use the download button below.")
            if st.session_state.get("generated_project_pdf"):
                st.download_button("Download PDF", data=st.session_state["generated_project_pdf"],
                                   file_name=f"AI_Smart_Monitoring_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                   mime="application/pdf", icon=":material/download:", width="stretch")

    ui.section("Report contents", "What the PDF will include right now")
    yes, no = ui.badge("Included", "normal"), ui.badge("Not run yet", "neutral")
    ui.render(ui.kv_panel([
        ("Sensor data summary & room-wise statistics", f"{len(ALL):,} readings (all data, unfiltered)"),
        ("Recent sensor readings", "Latest 12"),
        ("Sensor health simulation", ui.badge(HEALTH["status"].title(), "normal" if HEALTH["status"] == "HEALTHY" else "critical")),
        ("Edge vs cloud-style benchmark", yes if st.session_state.get("edge_cloud_benchmark") else no),
        ("Temperature & humidity forecast", yes if st.session_state.get("environment_forecast") else no),
        ("Prototype limitations", yes),
    ]))


def page_settings():
    header("Settings", "Integrations, model configuration and system information.", "Operations")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.section("Telegram notifications")
        on = be.telegram_configured()
        with st.container(border=True):
            ui.render(ui.badge("Connected" if on else "Not configured", "normal" if on else "neutral"))
            if on:
                st.caption("Bot token and chat ID are loaded from Streamlit secrets and never shown here.")
                if st.button("Send test message", icon=":material/send:", width="stretch"):
                    ok, detail = be.send_telegram_message(
                        "✅ Telegram integration test successful!\n"
                        "AI Smart Monitoring System is ready to send anomaly alerts.")
                    (st.success if ok else st.error)(detail)
            else:
                st.caption("Create `.streamlit/secrets.toml` in the project folder with:")
                st.code('TELEGRAM_BOT_TOKEN = "your-bot-token"\nTELEGRAM_CHAT_ID = "your-chat-id"', language="toml")
                st.caption("Restart the app afterwards. Never commit this file to GitHub.")

        ui.section("Severity rules")
        ui.render(ui.kv_panel([
            ("Normal", "Isolation Forest finds no anomaly"),
            ("Warning", "Anomaly within critical limits"),
            ("Critical", "Anomaly with temp ≥ 40 / ≤ 5 °C or humidity ≥ 95 / ≤ 10 %"),
            ("Explanations", "High temp > 35 °C · Low < 15 °C · High RH > 85 % · Low < 25 %"),
        ]))
    with c2:
        ui.section("AI model")
        ui.render(ui.kv_panel([
            ("Algorithm", "Isolation Forest (scikit-learn)"),
            ("Contamination", f"{be.MODEL_CONTAMINATION:.0%}"),
            ("Training samples", f"{be.MODEL_TRAINING_SAMPLES}"),
            ("Random state", f"{be.MODEL_RANDOM_STATE}"),
            ("Trees", f"{model.n_estimators}"),
        ]))
        ui.section("System")
        stats = be.database_stats()
        ui.render(ui.kv_panel([
            ("Database", f"<span style='font-family:var(--mono)'>{be.DB}</span> (SQLite)"),
            ("Stored readings", f"{stats['readings']:,}"),
            ("Telegram messages sent", f"{stats['telegram_sent']:,}"),
            ("First reading", ui.esc(stats["first"] or "—")),
            ("Latest reading", ui.esc(stats["last"] or "—")),
            ("Live refresh interval", f"{LIVE_INTERVAL_MS // 1000} s"),
            ("Hardware", "Simulated sensors (ESP32 / Raspberry Pi ready)"),
        ]))

    ui.section("How this prototype works")
    stages = [
        ("01", "IoT layer", "Temperature and humidity readings are simulated because physical sensor hardware is not connected yet."),
        ("02", "Edge processing", "Sensor readings are processed locally on this computer."),
        ("03", "AI layer", "The Isolation Forest algorithm identifies unusual sensor readings."),
        ("04", "Database", "SQLite stores readings, room identifiers, severity labels and processing times."),
        ("05", "Visualization", "Streamlit shows live values, analytics, alerts, benchmarks, forecasts and reports."),
    ]
    ui.render('<div class="pipeline">' + "".join(
        f'<div class="card"><div class="stage-n">{n}</div><div class="stage-t">{t}</div><div class="stage-d">{d}</div></div>'
        for n, t, d in stages) + "</div>")
    st.caption("Project status: software prototype · Hardware: physical IoT sensor not connected · "
               "Processing: local computer. Theme: dark (set in .streamlit/config.toml).")


# ==========================================
# 5. NAVIGATION
# ==========================================

pg = st.navigation({
    "Monitor": [
        st.Page(page_overview, title="Overview", icon=":material/space_dashboard:", default=True),
        st.Page(page_live, title="Live Monitoring", icon=":material/monitor_heart:", url_path="live"),
        st.Page(page_multisensor, title="Multi-Sensor Monitoring", icon=":material/sensors:", url_path="sensors"),
        st.Page(page_health, title="Sensor Health", icon=":material/health_and_safety:", url_path="health"),
    ],
    "Intelligence": [
        st.Page(page_anomaly, title="AI Anomaly Detection", icon=":material/neurology:", url_path="anomalies"),
        st.Page(page_benchmark, title="Edge vs Cloud Benchmark", icon=":material/speed:", url_path="benchmark"),
        st.Page(page_forecast, title="Forecast", icon=":material/insights:", url_path="forecast"),
    ],
    "Operations": [
        st.Page(page_alerts, title="Alert History", icon=":material/notifications:", url_path="alerts"),
        st.Page(page_reports, title="Reports & Exports", icon=":material/description:", url_path="reports"),
        st.Page(page_settings, title="Settings", icon=":material/settings:", url_path="settings"),
    ],
}, position="sidebar", expanded=True)


# ==========================================
# 6. SIDEBAR QUICK ACTIONS
# ==========================================

with st.sidebar:
    st.caption("QUICK ACTIONS")
    live_on = st.toggle("Live monitoring", value=False, key="live_on",
                        help=f"Adds a simulated reading every {LIVE_INTERVAL_MS // 1000} seconds.")
    st.button("30 sample readings", icon=":material/science:", on_click=action_generate_samples,
              width="stretch")
    st.button("40 multi-sensor readings", icon=":material/sensors:", on_click=action_generate_multi,
              width="stretch")
    st.caption("Simulated sensors · local edge AI · SQLite")


# ==========================================
# 7. LIVE READING + DATA LOAD (once per run, shared by all pages)
# ==========================================

LIVE = None
if live_on:
    st_autorefresh(interval=LIVE_INTERVAL_MS, key="live_monitoring_refresh")
    _room = st.session_state.get("sim_room", "Room 1")
    LIVE = be.take_live_reading(model, _room, be.SENSOR_MAP[_room])

# Load data after live monitoring saves the latest reading.
ALL = ui.with_ts(be.load_rows())
HEALTH = be.sensor_health(st.session_state.get("sensor_condition", "Healthy"))

for _msg, _icon in st.session_state.pop("_toasts", []):
    st.toast(_msg, icon=_icon)

_system_ok = HEALTH["status"] == "HEALTHY"
ui.top_bar(
    system_ok=_system_ok,
    system_label="All systems operational" if _system_ok else f"Sensor issue · {HEALTH['status'].title()}",
    live_on=live_on,
    last_reading=None if ALL.empty else str(ALL["timestamp"].iloc[-1]),
    rendered_at=datetime.now().strftime("%H:%M:%S"),
)

pg.run()
