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
    bmi = weight / ((height / 100) ** 2)

    st.header("Therapie")
    primary = st.selectbox("Primärwirkstoff", ["Propofol", "Remifentanil", "Fentanyl"])
    secondary_enabled = st.checkbox("Interaktion mit zweitem Wirkstoff", value=True)
    secondary = st.selectbox("Sekundärwirkstoff", ["Remifentanil", "Propofol", "Fentanyl"], index=0, disabled=not secondary_enabled)

    st.header("Zielwerte")
    target_type = st.radio("Regelgröße", ["Plasma", "Effektort"], horizontal=True)
    target_cp = st.number_input("Ziel-Plasma (mg/L)", 0.0, 100.0, 2.5 if primary == "Propofol" else 0.0, 0.1)
    target_ce = st.number_input("Ziel-Effektort (mg/L)", 0.0, 100.0, 2.0 if primary == "Propofol" else 0.0, 0.1)

    st.header("Applikation")
    profile_name = st.selectbox("Dosisprofil", ["Induktion", "Erhaltung", "Bolus + Infusion", "Individuell"])
    doses = DRUG_LIBRARY[primary]["profiles"][profile_name]
    dose_multiplier = st.number_input("Skalierungsfaktor", 0.1, 10.0, 1.0, 0.1)

    t_end = st.slider("Simulationsdauer (min)", 30, 720, 240, 10)

if secondary_enabled and secondary == primary:
    st.error("Primär- und Sekundärwirkstoff müssen verschieden sein.")
    st.stop()

validation = validate_inputs(primary, age, weight, height, sex, bmi, doses, t_end)
if validation:
    for msg in validation:
        st.warning(msg)

regimen = build_regimen(primary, secondary if secondary_enabled else None, profile_name, dose_multiplier)
result = simulate_regimen(primary, age, weight, height, sex, regimen, t_end, secondary_name=secondary if secondary_enabled else None)

cp = result["primary"]["cp"]
ce = result["primary"].get("ce")

a, b, c, d = st.columns(4)
a.metric("V1", f"{result['primary']['params']['v1']:.2f} L")
b.metric("CL1", f"{result['primary']['params']['cl1']:.2f} L/h")
c.metric("ke0", f"{result['primary']['params']['ke0']:.2f} 1/min")
d.metric("AUC", f"{result['primary']['auc']:.2f} mg·min/L")

fig = go.Figure()
fig.add_trace(go.Scatter(x=result['time_min'], y=cp, mode='lines', name=f"{primary} Plasma"))
if ce is not None:
    fig.add_trace(go.Scatter(x=result['time_min'], y=ce, mode='lines', name=f"{primary} Effektort"))
if secondary_enabled and result.get('secondary') is not None:
    fig.add_trace(go.Scatter(x=result['time_min'], y=result['secondary']['cp'], mode='lines', name=f"{secondary} Plasma", line=dict(dash='dot')))
fig.update_layout(template='plotly_white', height=620, xaxis_title='Zeit (min)', yaxis_title='Konzentration (mg/L)')
st.plotly_chart(fig, use_container_width=True)

st.subheader("Regime")
st.dataframe(pd.DataFrame(regimen), use_container_width=True)

st.subheader("Kennwerte")
metrics = pd.DataFrame(result['primary']['metrics'], index=[0])
st.dataframe(metrics, use_container_width=True)

csv_bytes = export_csv(result)
st.download_button("CSV exportieren", data=csv_bytes, file_name="anesthesia_pk_export.csv", mime="text/csv")

st.info("Plausibilitätsprüfung und Interaktionsmodell sind integriert; die App ist für Simulation und Lehre gedacht.")
