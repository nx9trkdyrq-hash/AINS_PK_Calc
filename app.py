import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from pk_models import (
    DRUG_LIBRARY,
    build_regimen,
    validate_inputs,
    simulate_regimen,
    export_csv,
)

st.set_page_config(page_title="Anästhesie PK Simulator", layout="wide")
st.title("Anästhesie PK Simulator")
st.caption("Mehrkompartimentmodell mit Effektort, Interaktion und Export.")

with st.sidebar:
    st.header("Patient")
    age = st.number_input("Alter (Jahre)", 0, 120, 50, 1)
    weight = st.number_input("Gewicht (kg)", 20.0, 300.0, 75.0, 0.5)
    height = st.number_input("Größe (cm)", 100.0, 230.0, 175.0, 0.5)
    sex = st.selectbox("Geschlecht", ["männlich", "weiblich"])

    st.header("Therapie")
    primary = st.selectbox("Primärwirkstoff", ["Propofol", "Remifentanil", "Fentanyl"])
    secondary_enabled = st.checkbox("Interaktion mit zweitem Wirkstoff", value=True)
    secondary = st.selectbox("Sekundärwirkstoff", ["Remifentanil", "Propofol", "Fentanyl"], index=0, disabled=not secondary_enabled)

    st.header("Applikation")
    profile_name = st.selectbox("Dosisprofil", ["Induktion", "Erhaltung", "Bolus + Infusion", "Individuell"])
    multiplier = st.number_input("Skalierungsfaktor", 0.1, 10.0, 1.0, 0.1)
    t_end = st.slider("Simulationsdauer (min)", 30, 720, 240, 10)

if secondary_enabled and secondary == primary:
    st.error("Primär- und Sekundärwirkstoff müssen verschieden sein.")
    st.stop()

regimen = build_regimen(primary, secondary if secondary_enabled else None, profile_name, multiplier)
warnings = validate_inputs(age, weight, height, sex, regimen, t_end)

for msg in warnings:
    st.warning(msg)

result = simulate_regimen(primary, age, weight, height, sex, regimen, t_end, secondary_name=secondary if secondary_enabled else None)

cp = result["primary"]["cp"]
ce = result["primary"]["ce"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("V1", f"{result['primary']['params']['v1']:.2f} L")
c2.metric("CL1", f"{result['primary']['params']['cl1']:.2f} L/h")
c3.metric("ke0", f"{result['primary']['params']['ke0']:.2f} 1/min")
c4.metric("AUC", f"{result['primary']['auc']:.2f} mg·min/L")

fig = go.Figure()
fig.add_trace(go.Scatter(x=result["time_min"], y=cp, mode="lines", name=f"{primary} Plasma"))
fig.add_trace(go.Scatter(x=result["time_min"], y=ce, mode="lines", name=f"{primary} Effektort"))

if secondary_enabled and result.get("secondary") is not None:
    fig.add_trace(go.Scatter(
        x=result["time_min"],
        y=result["secondary"]["cp"],
        mode="lines",
        name=f"{secondary} Plasma",
        line=dict(dash="dot"),
    ))

fig.update_layout(
    template="plotly_white",
    height=620,
    xaxis_title="Zeit (min)",
    yaxis_title="Konzentration",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Regime")
st.dataframe(pd.DataFrame(regimen), use_container_width=True)

st.subheader("Kennwerte")
st.dataframe(pd.DataFrame(result["primary"]["metrics"]), use_container_width=True)

st.download_button(
    label="CSV exportieren",
    data=export_csv(result),
    file_name="anesthesia_pk_export.csv",
    mime="text/csv",
)

st.info("Hinweis: Dieses Tool ist für Simulation und Lehre gedacht und ersetzt keine klinische Entscheidung.")
