import streamlit as st
import pandas as pd
import time
import random
from utils import generate_log, detect_anomaly

st.set_page_config(page_title="Log Anomaly Dashboard", layout="wide")

st.title("🚨 Log Anomaly Detection Dashboard (Demo)")

# Sidebar
st.sidebar.header("Controls")
speed = st.sidebar.slider("Stream Speed (seconds)", 1, 5, 2)

# Placeholder for logs
log_container = st.empty()
chart_container = st.empty()

data = []

while True:
    log = generate_log()
    is_anomaly = detect_anomaly(log)

    log["anomaly"] = is_anomaly
    data.append(log)

    df = pd.DataFrame(data[-50:])  # last 50 logs

    # Display logs
    with log_container.container():
        st.subheader("Live Logs")
        st.dataframe(df)

    # Chart
    with chart_container.container():
        st.subheader("Value Trend")
        st.line_chart(df["value"])

    # Alert
    if is_anomaly:
        st.error(f"🚨 Anomaly detected! Value: {log['value']}")

    time.sleep(speed)