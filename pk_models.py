import io
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

DRUG_LIBRARY = {
    "Propofol": {
        "model": "Schnider",
        "ke0": 0.26,
        "unit": "mg/L",
        "profiles": {
            "Induktion": [{"t0": 0, "t1": 0, "bolus_mg": 150, "infusion_mg_h": 0}],
            "Erhaltung": [{"t0": 0, "t1": 240, "bolus_mg": 0, "infusion_mg_h": 150}],
            "Bolus + Infusion": [
                {"t0": 0, "t1": 0, "bolus_mg": 120, "infusion_mg_h": 0},
                {"t0": 0, "t1": 240, "bolus_mg": 0, "infusion_mg_h": 120},
            ],
            "Individuell": [{"t0": 0, "t1": 0, "bolus_mg": 100, "infusion_mg_h": 0}],
        },
    },
    "Remifentanil": {
        "model": "Minto",
        "ke0": 1.2,
        "unit": "ng/mL",
        "profiles": {
            "Induktion": [{"t0": 0, "t1": 0, "bolus_mg": 1.0, "infusion_mg_h": 0}],
            "Erhaltung": [{"t0": 0, "t1": 240, "bolus_mg": 0, "infusion_mg_h": 0.25}],
            "Bolus + Infusion": [
                {"t0": 0, "t1": 0, "bolus_mg": 0.75, "infusion_mg_h": 0},
                {"t0": 0, "t1": 240, "bolus_mg": 0, "infusion_mg_h": 0.2},
            ],
            "Individuell": [{"t0": 0, "t1": 0, "bolus_mg": 0.5, "infusion_mg_h": 0}],
        },
    },
    "Fentanyl": {
        "model": "Generic",
        "ke0": 0.2,
        "unit": "ng/mL",
        "profiles": {
            "Induktion": [{"t0": 0, "t1": 0, "bolus_mg": 0.1, "infusion_mg_h": 0}],
            "Erhaltung": [{"t0": 0, "t1": 240, "bolus_mg": 0, "infusion_mg_h": 0.06}],
            "Bolus + Infusion": [
                {"t0": 0, "t1": 0, "bolus_mg": 0.1, "infusion_mg_h": 0},
                {"t0": 0, "t1": 240, "bolus_mg": 0, "infusion_mg_h": 0.05},
            ],
            "Individuell": [{"t0": 0, "t1": 0, "bolus_mg": 0.05, "infusion_mg_h": 0}],
        },
    },
}

def lbm(weight, height_cm, sex):
    if sex.lower().startswith("m"):
        return 1.1 * weight - 128 * (weight / height_cm) ** 2
    return 1.07 * weight - 148 * (weight / height_cm) ** 2


def propofol_params(age, weight, height_cm, sex):
    l = lbm(weight, height_cm, sex)
    return dict(v1=4.27, v2=max(5, 18.9 - 0.391 * (age - 53)), v3=238, cl1=max(0.2, 1.89 + 0.0456 * (weight - 77) - 0.0681 * (l - 59)), cl2=max(0.05, 1.29 - 0.024 * (age - 53)), cl3=0.836, ke0=0.26)


def remifentanil_params(age, weight, height_cm, sex):
    l = lbm(weight, height_cm, sex)
    return dict(v1=max(1.0, 5.1 - 0.0201 * (age - 40) + 0.072 * (l - 55)), v2=max(2.0, 9.82 - 0.0811 * (age - 40) + 0.108 * (l - 55)), v3=5.42, cl1=max(0.2, 2.6 - 0.0162 * (age - 40) + 0.0191 * (l - 55)), cl2=max(0.1, 2.05 - 0.0301 * (age - 40) + 0.0041 * (l - 55)), cl3=0.076, ke0=1.2)


def fentanyl_params(age, weight, height_cm, sex):
    return dict(v1=max(2.0, 4.0 + 0.05 * weight), v2=max(5.0, 10.0 + 0.1 * weight), v3=max(50.0, 100.0 + 0.2 * weight), cl1=max(0.1, 0.5 + 0.01 * weight), cl2=0.4, cl3=0.1, ke0=0.2)


def get_params(drug, age, weight, height_cm, sex):
    if drug == "Propofol":
        return propofol_params(age, weight, height_cm, sex)
    if drug == "Remifentanil":
        return remifentanil_params(age, weight, height_cm, sex)
    return fentanyl_params(age, weight, height_cm, sex)


def build_regimen(primary, secondary, profile_name, multiplier=1.0):
    regimen = []
    for item in DRUG_LIBRARY[primary]["profiles"][profile_name]:
        x = item.copy()
        x["drug"] = primary
        x["bolus_mg"] *= multiplier
        x["infusion_mg_h"] *= multiplier
        regimen.append(x)
    if secondary:
        sec_profile = profile_name if profile_name in DRUG_LIBRARY[secondary]["profiles"] else "Erhaltung"
        for item in DRUG_LIBRARY[secondary]["profiles"][sec_profile]:
            x = item.copy()
            x["drug"] = secondary
            x["bolus_mg"] *= multiplier
            x["infusion_mg_h"] *= multiplier
            regimen.append(x)
    return regimen


def validate_inputs(age, weight, height_cm, sex, regimen, t_end):
    msgs = []
    bmi = weight / ((height_cm / 100) ** 2)
    if not (12 <= age <= 100):
        msgs.append("Alter außerhalb des typischen Erwachsenenbereichs.")
    if not (40 <= weight <= 200):
        msgs.append("Gewicht außerhalb des üblichen Modellbereichs.")
    if not (140 <= height_cm <= 210):
        msgs.append("Größe außerhalb des üblichen Modellbereichs.")
    if not (15 <= bmi <= 45):
        msgs.append("BMI außerhalb des plausiblen Bereichs.")
    if t_end < 30:
        msgs.append("Simulationsdauer ist sehr kurz.")
    if any(r["bolus_mg"] < 0 or r["infusion_mg_h"] < 0 for r in regimen):
        msgs.append("Dosisprofile enthalten negative Werte.")
    return msgs


def _simulate_single(drug, age, weight, height_cm, sex, regimen, t_end):
    p = get_params(drug, age, weight, height_cm, sex)
    v1, v2, v3 = p["v1"], p["v2"], p["v3"]
    cl1, cl2, cl3 = p["cl1"], p["cl2"], p["cl3"]
    ke0 = p["ke0"]
    k10 = cl1 / v1
    k12 = cl2 / v1
    k13 = cl3 / v1
    k21 = cl2 / v2
    k31 = cl3 / v3
    intervals = []
    for r in regimen:
        if r["drug"] == drug:
            if r["bolus_mg"] > 0:
                intervals.append((0, 0, r["bolus_mg"]))
            if r["infusion_mg_h"] > 0:
                intervals.append((r["t0"], r["t1"], r["infusion_mg_h"] / 60.0))
    def rin(t):
        s = 0.0
        for t0, t1, rate in intervals:
            if t0 <= t <= t1:
                s += rate
        return s
    def ode(t, y):
        a1, a2, a3, ce = y
        da1 = rin(t) - (k10 + k12 + k13) * a1 + k21 * a2 + k31 * a3
        da2 = k12 * a1 - k21 * a2
        da3 = k13 * a1 - k31 * a3
        dce = ke0 * (a1 / v1 - ce)
        return [da1, da2, da3, dce]
    y0 = [0.0, 0.0, 0.0, 0.0]
    t_eval = np.linspace(0, t_end, 1200)
    sol = solve_ivp(ode, [0, t_end], y0, t_eval=t_eval, method="LSODA")
    cp = sol.y[0] / v1
    ce = sol.y[3]
    auc = np.trapz(cp, sol.t)
    cmax = cp.max()
    tmax = sol.t[cp.argmax()]
    return {"params": p, "cp": cp, "ce": ce, "auc": auc, "metrics": {"Cmax mg/L": [cmax], "Tmax min": [tmax], "AUC mg*min/L": [auc], "V1 L": [v1], "CL1 L/h": [cl1]}}


def interaction_index(primary, secondary, cp_p, cp_s):
    if primary == "Propofol" and secondary in ["Remifentanil", "Fentanyl"]:
        return 1 / (1 + np.exp(-0.7 * (cp_s - 0.8)))
    if primary in ["Remifentanil", "Fentanyl"] and secondary == "Propofol":
        return 1 / (1 + np.exp(-0.8 * (cp_s - 1.5)))
    return np.zeros_like(cp_p)


def simulate_regimen(primary, age, weight, height_cm, sex, regimen, t_end, secondary_name=None):
    primary_sim = _simulate_single(primary, age, weight, height_cm, sex, regimen, t_end)
    out = {"time_min": np.linspace(0, t_end, 1200), "primary": primary_sim}
    if secondary_name:
        secondary_sim = _simulate_single(secondary_name, age, weight, height_cm, sex, regimen, t_end)
        out["secondary"] = secondary_sim
        out["interaction_index"] = interaction_index(primary, secondary_name, primary_sim["cp"], secondary_sim["cp"])
    return out


def export_csv(result):
    df = pd.DataFrame({"time_min": result["time_min"], "primary_cp": result["primary"]["cp"], "primary_ce": result["primary"]["ce"]})
    if result.get("secondary") is not None:
        df["secondary_cp"] = result["secondary"]["cp"]
        df["secondary_ce"] = result["secondary"]["ce"]
    if result.get("interaction_index") is not None:
        df["interaction_index"] = result["interaction_index"]
    return df.to_csv(index=False).encode("utf-8")
