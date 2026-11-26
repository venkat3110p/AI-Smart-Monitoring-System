"""
Design system for the Smart Monitoring dashboard.

It holds the colour tokens, the global CSS, inline SVG icons, the reusable
HTML components (KPI cards, badges, sensor cards, alert rows) and a shared
Plotly theme. Every page draws from here, so the look stays consistent.
"""

import html

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


# ==========================================
# DESIGN TOKENS
# ==========================================

BG = "#0A0E17"
SURFACE = "#111725"
BORDER = "rgba(255,255,255,0.07)"
TEXT = "#E7EBF3"
MUTED = "#8B95A9"
FAINT = "#5B6578"

ACCENT = "#7C8CFF"      # brand / temperature series
TEAL = "#4FD1C5"        # humidity series
GREEN = "#3DD68C"       # Normal
AMBER = "#F5B544"       # Warning
RED = "#F7616A"         # Critical

TONES = {
    "accent": ACCENT, "teal": TEAL, "normal": GREEN, "warning": AMBER,
    "critical": RED, "neutral": MUTED,
}
SEVERITY_TONE = {"Normal": "normal", "Warning": "warning", "Critical": "critical"}
SEVERITY_COLOR = {"Normal": GREEN, "Warning": AMBER, "Critical": RED}


def esc(value):
    return html.escape(str(value))


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ==========================================
# ICONS (inline SVG, Lucide-style, no extra dependency)
# ==========================================

_ICON_PATHS = {
    "activity": ['<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>'],
    "thermometer": ['<path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/>'],
    "droplet": ['<path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/>'],
    "alert": ['<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>',
              '<path d="M12 9v4"/>', '<path d="M12 17h.01"/>'],
    "critical": ['<path d="M7.86 2h8.28L22 7.86v8.28L16.14 22H7.86L2 16.14V7.86L7.86 2z"/>',
                 '<path d="M12 8v4"/>', '<path d="M12 16h.01"/>'],
    "check": ['<circle cx="12" cy="12" r="10"/>', '<path d="m9 12 2 2 4-4"/>'],
    "pin": ['<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/>', '<circle cx="12" cy="10" r="3"/>'],
    "database": ['<ellipse cx="12" cy="5" rx="9" ry="3"/>', '<path d="M3 5v14a9 3 0 0 0 18 0V5"/>',
                 '<path d="M3 12a9 3 0 0 0 18 0"/>'],
    "zap": ['<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>'],
    "cloud": ['<path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>'],
    "cpu": ['<rect x="4" y="4" width="16" height="16" rx="2"/>', '<rect x="9" y="9" width="6" height="6"/>',
            '<path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>'],
    "radio": ['<path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9"/>', '<path d="M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5"/>',
              '<circle cx="12" cy="12" r="2"/>', '<path d="M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5"/>',
              '<path d="M19.1 4.9C23 8.8 23 15.1 19.1 19"/>'],
    "clock": ['<circle cx="12" cy="12" r="10"/>', '<path d="M12 6v6l4 2"/>'],
    "bell": ['<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/>', '<path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>'],
    "sparkles": ['<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/>'],
    "trend": ['<path d="M22 7 13.5 15.5 8.5 10.5 2 17"/>', '<path d="M16 7h6v6"/>'],
    "file": ['<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>', '<path d="M14 2v4a2 2 0 0 0 2 2h4"/>'],
    "send": ['<path d="m22 2-7 20-4-9-9-4Z"/>', '<path d="M22 2 11 13"/>'],
    "layers": ['<path d="m12 2 10 5-10 5L2 7l10-5Z"/>', '<path d="m2 17 10 5 10-5"/>', '<path d="m2 12 10 5 10-5"/>'],
}


def icon(name, size=16, color="currentColor", stroke=1.8):
    paths = "".join(_ICON_PATHS.get(name, _ICON_PATHS["activity"]))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
            f'stroke-linejoin="round" style="flex-shrink:0">{paths}</svg>')


# ==========================================
# GLOBAL CSS
# ==========================================

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root {{
  --bg:{BG}; --surface:{SURFACE}; --border:{BORDER}; --border-strong:rgba(255,255,255,0.12);
  --text:{TEXT}; --muted:{MUTED}; --faint:{FAINT}; --accent:{ACCENT};
  --green:{GREEN}; --amber:{AMBER}; --red:{RED};
  --radius:14px; --mono:'JetBrains Mono', ui-monospace, monospace;
}}

html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea, select {{
  font-family:'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
.stApp {{
  background:
    radial-gradient(1100px 520px at 18% -8%, {rgba(ACCENT, 0.09)}, transparent 62%),
    radial-gradient(900px 480px at 100% 0%, {rgba(TEAL, 0.05)}, transparent 60%),
    var(--bg);
  color:var(--text);
}}
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:1.4rem; padding-bottom:4rem; max-width:1440px; }}
[data-testid="stMainBlockContainer"] {{ padding-left:2.2rem; padding-right:2.2rem; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
  background:linear-gradient(180deg, #0D1220 0%, #0A0E17 100%);
  border-right:1px solid var(--border);
}}
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] {{ padding-top:.25rem; }}
section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"],
section[data-testid="stSidebar"] header {{
  color:var(--faint) !important; font-size:.68rem !important; font-weight:600 !important;
  letter-spacing:.12em; text-transform:uppercase;
}}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {{
  border-radius:9px; margin:1px 0; transition:background .15s ease, color .15s ease;
}}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] span {{ color:var(--muted); font-weight:500; font-size:.88rem; }}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {{ background:rgba(255,255,255,0.04); }}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {{
  background:{rgba(ACCENT, 0.12)}; box-shadow:inset 0 0 0 1px {rgba(ACCENT, 0.25)};
}}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color:var(--text); }}

/* ---------- native widgets ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stPopover"] button {{
  border-radius:10px; border:1px solid var(--border-strong); background:rgba(255,255,255,0.03);
  color:var(--text); font-weight:500; transition:all .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stPopover"] button:hover {{
  border-color:{rgba(ACCENT, 0.6)}; background:{rgba(ACCENT, 0.08)}; color:var(--text);
}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background:linear-gradient(180deg, #8B99FF 0%, #6C7BF5 100%); border:1px solid {rgba(ACCENT, 0.9)};
  color:#0B0F1C; font-weight:600; box-shadow:0 1px 0 rgba(255,255,255,0.25) inset, 0 6px 18px {rgba(ACCENT, 0.25)};
}}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {{
  color:#0B0F1C; filter:brightness(1.07); background:linear-gradient(180deg, #8B99FF 0%, #6C7BF5 100%);
}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border:1px solid var(--border); border-radius:12px; overflow:hidden;
}}
[data-testid="stPlotlyChart"] {{
  background:linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
  border:1px solid var(--border); border-radius:var(--radius); padding:10px 12px 4px;
}}
[data-testid="stExpander"] details {{ border:1px solid var(--border); border-radius:12px; background:rgba(255,255,255,0.02); }}
[data-testid="stAlert"] {{ border-radius:12px; }}
[data-testid="stWidgetLabel"] p {{ color:var(--muted); font-size:.82rem; font-weight:500; }}
hr {{ border-color:var(--border) !important; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{ border-radius:10px; }}

/* ---------- top bar ---------- */
.topbar {{
  display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  padding:10px 14px 10px 12px; margin-bottom:22px;
  border:1px solid var(--border); border-radius:16px;
  background:rgba(17,23,37,0.65); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
}}
.brand {{ display:flex; align-items:center; gap:12px; min-width:0; }}
.logo {{
  width:36px; height:36px; border-radius:10px; display:grid; place-items:center; flex-shrink:0;
  background:linear-gradient(135deg, {ACCENT} 0%, {TEAL} 100%); color:#0B0F1C;
  box-shadow:0 4px 14px {rgba(ACCENT, 0.35)};
}}
.brand-name {{ font-weight:600; font-size:.95rem; color:var(--text); letter-spacing:-.01em; line-height:1.2; }}
.brand-sub {{ font-size:.74rem; color:var(--faint); }}
.topbar-right {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
.stamp {{ display:flex; align-items:center; gap:6px; color:var(--faint); font-size:.76rem; font-family:var(--mono); padding:0 4px; }}

/* ---------- badges ---------- */
.badge {{
  display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px;
  font-size:.72rem; font-weight:600; letter-spacing:.02em; white-space:nowrap; border:1px solid;
}}
.badge .dot {{ width:6px; height:6px; border-radius:50%; background:currentColor; }}
.badge.live .dot {{ animation:pulse 1.8s ease-in-out infinite; box-shadow:0 0 0 0 currentColor; }}
@keyframes pulse {{ 0% {{ box-shadow:0 0 0 0 {rgba(GREEN, 0.6)}; }} 70% {{ box-shadow:0 0 0 6px {rgba(GREEN, 0)}; }} 100% {{ box-shadow:0 0 0 0 {rgba(GREEN, 0)}; }} }}

/* ---------- page title ---------- */
.page-head {{ margin:2px 0 18px; }}
.eyebrow {{ color:var(--accent); font-size:.7rem; font-weight:600; letter-spacing:.14em; text-transform:uppercase; }}
.page-title {{ font-size:1.65rem; font-weight:650; letter-spacing:-.02em; color:var(--text); margin:4px 0 4px; line-height:1.2; }}
.page-sub {{ color:var(--muted); font-size:.92rem; max-width:760px; }}

.section {{ display:flex; align-items:flex-end; justify-content:space-between; gap:12px; margin:26px 0 12px; }}
.section h4 {{ margin:0; padding:0; font-size:1rem; font-weight:600; color:var(--text); letter-spacing:-.01em; }}
.section p {{ margin:2px 0 0; color:var(--muted); font-size:.83rem; }}

/* ---------- cards ---------- */
.grid {{ display:grid; gap:14px; margin-bottom:6px; }}
.grid.kpis {{ grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); }}
.grid.sensors {{ grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); }}
.grid.three {{ grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); }}

.card {{
  position:relative; background:linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015));
  border:1px solid var(--border); border-radius:var(--radius); padding:16px 18px;
  transition:border-color .2s ease, transform .2s ease, background .2s ease;
}}
.card:hover {{ border-color:var(--border-strong); transform:translateY(-1px); }}
.kpi-top {{ display:flex; align-items:center; justify-content:space-between; }}
.kpi-label {{ color:var(--muted); font-size:.8rem; font-weight:500; }}
.kpi-icon {{ width:30px; height:30px; border-radius:8px; display:grid; place-items:center; }}
.kpi-value {{ font-size:1.9rem; font-weight:600; letter-spacing:-.02em; color:var(--text); margin-top:10px; line-height:1.1; font-variant-numeric:tabular-nums; }}
.kpi-value small {{ font-size:.95rem; color:var(--muted); font-weight:500; margin-left:3px; }}
.kpi-sub {{ color:var(--faint); font-size:.76rem; margin-top:6px; }}
.kpi-accent {{ position:absolute; left:18px; right:18px; top:0; height:1px; }}

.sensor-head {{ display:flex; align-items:center; justify-content:space-between; gap:8px; }}
.sensor-name {{ font-weight:600; font-size:.95rem; color:var(--text); }}
.sensor-id {{ font-family:var(--mono); font-size:.72rem; color:var(--faint); margin-top:2px; }}
.sensor-vals {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:16px 0 14px; }}
.sensor-val .lbl {{ display:flex; align-items:center; gap:5px; color:var(--muted); font-size:.74rem; }}
.sensor-val .num {{ font-size:1.45rem; font-weight:600; color:var(--text); margin-top:3px; font-variant-numeric:tabular-nums; }}
.sensor-val .num small {{ font-size:.8rem; color:var(--muted); font-weight:500; margin-left:2px; }}
.sensor-foot {{ display:flex; justify-content:space-between; gap:8px; white-space:nowrap; border-top:1px solid var(--border); padding-top:10px; color:var(--faint); font-size:.74rem; }}
.sensor-foot span {{ display:inline-flex; align-items:center; gap:4px; }}
.sensor-foot b {{ color:var(--muted); font-weight:600; }}

.alert-row {{
  display:grid; grid-template-columns:auto 1fr auto; gap:14px; align-items:center;
  padding:12px 14px; border:1px solid var(--border); border-radius:12px; margin-bottom:8px;
  background:rgba(255,255,255,0.02); transition:border-color .15s ease;
}}
.alert-row:hover {{ border-color:var(--border-strong); }}
.alert-ico {{ width:32px; height:32px; border-radius:9px; display:grid; place-items:center; }}
.alert-title {{ font-weight:600; font-size:.88rem; color:var(--text); }}
.alert-body {{ color:var(--muted); font-size:.8rem; margin-top:2px; }}
.alert-meta {{ text-align:right; color:var(--faint); font-size:.74rem; font-family:var(--mono); }}

.kv {{ display:flex; justify-content:space-between; gap:12px; padding:9px 0; border-bottom:1px solid var(--border); font-size:.85rem; }}
.kv:last-child {{ border-bottom:none; }}
.kv span {{ color:var(--muted); }}
.kv b {{ color:var(--text); font-weight:500; text-align:right; }}

.callout {{
  display:flex; gap:10px; align-items:flex-start; padding:11px 14px; border-radius:12px;
  font-size:.82rem; line-height:1.5; border:1px solid; margin:8px 0 4px;
}}
.empty {{ text-align:center; padding:44px 20px; border:1px dashed var(--border-strong); border-radius:var(--radius); color:var(--muted); }}
.empty .t {{ color:var(--text); font-weight:600; margin:10px 0 4px; }}

.pipeline {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; }}
.stage-n {{ font-family:var(--mono); font-size:.7rem; color:var(--accent); }}
.stage-t {{ font-weight:600; margin:6px 0 4px; color:var(--text); font-size:.92rem; }}
.stage-d {{ color:var(--muted); font-size:.8rem; line-height:1.5; }}

@media (max-width: 1200px) {{
  [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; row-gap:1rem; }}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    flex:1 1 320px !important; min-width:min(320px, 100%) !important; width:auto !important;
  }}
}}
@media (max-width: 900px) {{
  [data-testid="stMainBlockContainer"] {{ padding-left:1rem; padding-right:1rem; }}
  .page-title {{ font-size:1.35rem; }}
  .kpi-value {{ font-size:1.6rem; }}
  .alert-row {{ grid-template-columns:auto 1fr; }}
  .alert-meta {{ grid-column:2; text-align:left; }}
}}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def render(html_str):
    """Render a block of trusted component HTML."""
    st.markdown(html_str, unsafe_allow_html=True)


# ==========================================
# COMPONENTS
# ==========================================

def badge(text, tone="neutral", live=False, dot=True):
    c = TONES.get(tone, MUTED)
    cls = "badge live" if live else "badge"
    d = '<span class="dot"></span>' if dot else ""
    return (f'<span class="{cls}" style="color:{c};border-color:{rgba(c, 0.32)};'
            f'background:{rgba(c, 0.09)}">{d}{esc(text)}</span>')


def severity_badge(severity):
    return badge(severity, SEVERITY_TONE.get(severity, "neutral"))


def top_bar(system_ok, system_label, live_on, last_reading, rendered_at):
    status = badge(system_label, "normal" if system_ok else "warning")
    live = badge("Live" if live_on else "Paused", "normal" if live_on else "neutral", live=live_on)
    last = esc(last_reading) if last_reading else "no data yet"
    render(
        '<div class="topbar">'
        f'<div class="brand"><div class="logo">{icon("activity", 19, "#0B0F1C", 2.2)}</div>'
        '<div><div class="brand-name">AI Smart Monitoring</div>'
        '<div class="brand-sub">IoT &amp; Edge Computing · Isolation Forest</div></div></div>'
        f'<div class="topbar-right">{status}{live}'
        f'<span class="stamp" title="Most recent reading in the database">{icon("clock", 13)} last reading {last}</span>'
        f'<span class="stamp" title="Page refreshed at">{icon("activity", 13)} {esc(rendered_at)}</span>'
        '</div></div>'
    )


def page_title(title, subtitle="", eyebrow=""):
    eb = f'<div class="eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    sub = f'<div class="page-sub">{esc(subtitle)}</div>' if subtitle else ""
    render(f'<div class="page-head">{eb}<div class="page-title">{esc(title)}</div>{sub}</div>')


def section(title, caption="", right_html=""):
    cap = f"<p>{esc(caption)}</p>" if caption else ""
    render(f'<div class="section"><div><h4>{esc(title)}</h4>{cap}</div>{right_html}</div>')


def kpi_card(label, value, sub="", icon_name="activity", tone="accent", unit=""):
    c = TONES.get(tone, ACCENT)
    u = f"<small>{esc(unit)}</small>" if unit else ""
    return (
        '<div class="card">'
        f'<div class="kpi-accent" style="background:linear-gradient(90deg, transparent, {rgba(c, 0.7)}, transparent)"></div>'
        f'<div class="kpi-top"><span class="kpi-label">{esc(label)}</span>'
        f'<span class="kpi-icon" style="background:{rgba(c, 0.12)};color:{c}">{icon(icon_name, 16)}</span></div>'
        f'<div class="kpi-value">{esc(value)}{u}</div>'
        f'<div class="kpi-sub">{esc(sub)}</div></div>'
    )


def kpi_grid(cards, kind="kpis"):
    render(f'<div class="grid {kind}">{"".join(cards)}</div>')


def sensor_card(room, sensor_id, temperature, humidity, severity, anomalies, readings, last_ts):
    sev = severity or "No data"
    b = severity_badge(sev) if severity else badge("No data", "neutral")
    t = f"{temperature:.1f}" if temperature is not None else "—"
    h = f"{humidity:.1f}" if humidity is not None else "—"
    return (
        '<div class="card">'
        f'<div class="sensor-head"><div><div class="sensor-name">{esc(room)}</div>'
        f'<div class="sensor-id">{esc(sensor_id)}</div></div>{b}</div>'
        '<div class="sensor-vals">'
        f'<div class="sensor-val"><div class="lbl">{icon("thermometer", 13)} Temperature</div><div class="num">{t}<small>°C</small></div></div>'
        f'<div class="sensor-val"><div class="lbl">{icon("droplet", 13)} Humidity</div><div class="num">{h}<small>%</small></div></div>'
        '</div>'
        f'<div class="sensor-foot"><span><b>{anomalies}</b> anomalies · {readings} readings</span>'
        f'<span title="{esc(last_ts or "")}">{icon("clock", 11)} {esc(str(last_ts)[11:19]) if last_ts else "—"}</span></div></div>'
    )


def alert_row(title, body, meta_top, meta_bottom, severity):
    c = SEVERITY_COLOR.get(severity, MUTED)
    ic = "critical" if severity == "Critical" else "alert" if severity == "Warning" else "check"
    return (
        '<div class="alert-row">'
        f'<div class="alert-ico" style="background:{rgba(c, 0.12)};color:{c}">{icon(ic, 16)}</div>'
        f'<div><div class="alert-title">{esc(title)} &nbsp;{severity_badge(severity)}</div>'
        f'<div class="alert-body">{esc(body)}</div></div>'
        f'<div class="alert-meta">{esc(meta_top)}<br>{esc(meta_bottom)}</div></div>'
    )


def kv_panel(items):
    rows = "".join(f'<div class="kv"><span>{esc(k)}</span><b>{v}</b></div>' for k, v in items)
    return f'<div class="card">{rows}</div>'


def callout(text, tone="neutral", icon_name="layers"):
    c = TONES.get(tone, MUTED)
    render(f'<div class="callout" style="color:{TEXT if tone == "neutral" else c};'
           f'border-color:{rgba(c, 0.28)};background:{rgba(c, 0.07)}">'
           f'<span style="color:{c};margin-top:1px">{icon(icon_name, 15)}</span><span>{esc(text)}</span></div>')


def empty_state(title, body, icon_name="database"):
    render(f'<div class="empty"><div style="color:{FAINT}">{icon(icon_name, 28)}</div>'
           f'<div class="t">{esc(title)}</div><div>{esc(body)}</div></div>')


# ==========================================
# PLOTLY THEME + CHART BUILDERS
# ==========================================

_GRID = "rgba(255,255,255,0.06)"

pio.templates["smartsense"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=MUTED, size=12),
    colorway=[ACCENT, TEAL, GREEN, AMBER, RED, "#B794F6"],
    title=dict(font=dict(size=14, color=TEXT), x=0.01, xanchor="left", y=0.97, yanchor="top"),
    xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID, tickcolor=_GRID, showgrid=False),
    yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, linecolor="rgba(0,0,0,0)", tickcolor="rgba(0,0,0,0)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                font=dict(color=MUTED, size=11), title_text=""),
    margin=dict(l=8, r=20, t=76, b=36),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor="rgba(255,255,255,0.15)", font=dict(color=TEXT, size=12)),
    hovermode="x unified",
))
pio.templates.default = "plotly_dark+smartsense"

_CHART_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def show_chart(fig, key=None):
    st.plotly_chart(fig, width="stretch", theme=None, config=_CHART_CONFIG, key=key)


def history_chart(df, column, color, title, unit, bands=None, height=320):
    """Single-metric history with anomaly markers and optional threshold bands."""
    d = df.sort_values("ts")
    fig = go.Figure()
    for lo, hi, tone_color in (bands or []):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=rgba(tone_color, 0.06), line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=d["ts"], y=d[column], mode="lines", name=title,
        line=dict(color=color, width=2, shape="linear"),
        fill="tozeroy", fillcolor=rgba(color, 0.07),
        hovertemplate=f"%{{y:.1f}} {unit}<extra></extra>",
    ))
    ab = d[d["anomaly"] == 1]
    if not ab.empty:
        fig.add_trace(go.Scatter(
            x=ab["ts"], y=ab[column], mode="markers", name="Anomaly",
            marker=dict(color=[SEVERITY_COLOR.get(s, RED) for s in ab["severity"]], size=9,
                        line=dict(color=BG, width=1.5)),
            customdata=ab[["room_id", "severity"]].to_numpy(),
            hovertemplate="%{customdata[1]} · %{customdata[0]}<extra></extra>",
        ))
    lo = min(0, float(d[column].min()) - 2) if not d.empty else 0
    fig.update_layout(title=title, height=height, showlegend=False, margin=dict(t=52))
    fig.update_yaxes(ticksuffix=f" {unit}", rangemode="tozero" if lo == 0 else "normal")
    return fig


def dual_history_chart(df, height=360):
    from plotly.subplots import make_subplots
    d = df.sort_values("ts")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=d["ts"], y=d["temperature"], name="Temperature", mode="lines",
                             line=dict(color=ACCENT, width=2, shape="linear"),
                             hovertemplate="%{y:.1f} °C"), secondary_y=False)
    fig.add_trace(go.Scatter(x=d["ts"], y=d["humidity"], name="Humidity", mode="lines",
                             line=dict(color=TEAL, width=2, shape="linear"),
                             hovertemplate="%{y:.1f} %"), secondary_y=True)
    ab = d[d["anomaly"] == 1]
    if not ab.empty:
        fig.add_trace(go.Scatter(x=ab["ts"], y=ab["temperature"], name="Anomaly", mode="markers",
                                 marker=dict(color=[SEVERITY_COLOR.get(s, RED) for s in ab["severity"]],
                                             size=9, line=dict(color=BG, width=1.5)),
                                 hovertemplate="%{y:.1f} °C anomaly"), secondary_y=False)
    fig.update_layout(title="Temperature & humidity over time", height=height)
    fig.update_yaxes(title_text=None, ticksuffix=" °C", secondary_y=False)
    fig.update_yaxes(title_text=None, ticksuffix=" %", secondary_y=True, showgrid=False, range=[0, 105], dtick=20)
    return fig


def gauge(value, title, max_value, bands, unit, bar_color, height=230):
    steps = [dict(range=[lo, hi], color=rgba(c, 0.16)) for lo, hi, c in bands]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(suffix=unit, font=dict(size=34, color=TEXT), valueformat=".1f"),
        title=dict(text=title, font=dict(size=13, color=MUTED)),
        gauge=dict(
            axis=dict(range=[0, max_value], tickcolor=FAINT, tickfont=dict(color=FAINT, size=10), tickwidth=1),
            bar=dict(color=bar_color, thickness=0.22),
            bgcolor="rgba(0,0,0,0)", borderwidth=0, steps=steps,
        ),
    ))
    fig.update_layout(height=height, margin=dict(l=28, r=28, t=48, b=8))
    return fig


def severity_donut(df, height=300):
    counts = df["severity"].value_counts().reindex(["Normal", "Warning", "Critical"], fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.7, sort=False, direction="clockwise",
        marker=dict(colors=[GREEN, AMBER, RED], line=dict(color=BG, width=3)),
        textinfo="none", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title="Severity distribution", height=height, hovermode=False,
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5),
        annotations=[dict(text=f"<b style='font-size:24px;color:{TEXT}'>{int(counts.sum())}</b><br>readings",
                          showarrow=False, font=dict(size=11, color=MUTED))],
        margin=dict(l=8, r=8, t=52, b=30),
    )
    return fig


def bar_chart(categories, values, title, unit="", colors=None, height=300, text_fmt="{:.1f}"):
    fig = go.Figure(go.Bar(
        x=list(categories), y=list(values),
        marker=dict(color=colors or ACCENT, line=dict(width=0), cornerradius=6),
        text=[text_fmt.format(v) + unit for v in values], textposition="outside",
        textfont=dict(color=TEXT, size=11), cliponaxis=False,
        hovertemplate="%{x}: %{y:.3f}" + unit + "<extra></extra>",
    ))
    fig.update_layout(title=title, height=height, bargap=0.45, hovermode="closest", margin=dict(t=56))
    fig.update_yaxes(ticksuffix=unit)
    return fig


def scatter_temp_humidity(df, height=380):
    fig = go.Figure()
    for sev in ["Normal", "Warning", "Critical"]:
        part = df[df["severity"] == sev]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["temperature"], y=part["humidity"], mode="markers", name=sev,
            marker=dict(color=SEVERITY_COLOR[sev], size=8 if sev == "Normal" else 11,
                        opacity=0.65 if sev == "Normal" else 0.95, line=dict(color=BG, width=1)),
            customdata=part[["room_id", "timestamp"]].to_numpy(),
            hovertemplate="%{x:.1f} °C · %{y:.1f}%<br>%{customdata[0]} · %{customdata[1]}<extra>" + sev + "</extra>",
        ))
    # Learned normal region from the training distribution (mean ± 2σ).
    fig.add_shape(type="rect", x0=27 - 2 * 2.2, x1=27 + 2 * 2.2, y0=55 - 2 * 8, y1=55 + 2 * 8,
                  line=dict(color=rgba(ACCENT, 0.5), width=1, dash="dot"), fillcolor=rgba(ACCENT, 0.04), layer="below")
    fig.add_annotation(x=27 + 2 * 2.2, y=55 + 2 * 8, text="learned normal region", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(color=ACCENT, size=10))
    fig.update_layout(title="Temperature vs humidity", height=height, hovermode="closest", margin=dict(b=48))
    fig.update_xaxes(title_text="Temperature (°C)", showgrid=True)
    fig.update_yaxes(title_text="Humidity (%)")
    return fig


def forecast_chart(times, actual, predicted, future_times, future_values, title, unit, color, height=330):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=actual, name="Actual", mode="lines+markers",
                             line=dict(color=color, width=2), marker=dict(size=5),
                             hovertemplate=f"%{{y:.2f}} {unit}"))
    fig.add_trace(go.Scatter(x=times, y=predicted, name="Back-test prediction", mode="lines",
                             line=dict(color=MUTED, width=1.5, dash="dot"), connectgaps=False,
                             hovertemplate=f"%{{y:.2f}} {unit}"))
    if len(times):
        fig.add_trace(go.Scatter(x=[times[-1]] + list(future_times), y=[actual[-1]] + list(future_values),
                                 name="Forecast (next 5 min)", mode="lines+markers",
                                 line=dict(color=AMBER, width=2.5), marker=dict(size=7, symbol="diamond"),
                                 hovertemplate=f"%{{y:.2f}} {unit}"))
        fig.add_vrect(x0=times[-1], x1=future_times[-1], fillcolor=rgba(AMBER, 0.05), line_width=0, layer="below")
    fig.update_layout(title=title, height=height)
    fig.update_yaxes(ticksuffix=f" {unit}")
    return fig


def with_ts(df):
    out = df.copy()
    out["ts"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out
