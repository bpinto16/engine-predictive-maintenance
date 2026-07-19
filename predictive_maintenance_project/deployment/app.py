import numpy as np
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Engine Maintenance Predictor",
    layout="wide",
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
HF_MODEL_REPO  = "bpinto16/predictive-maintenance-mlflow-model"
MODEL_FILENAME = "best_predictive_maintenance_mlflow_model.joblib"
RANGES_REPO    = "bpinto16/Predictive-Maintenance-HFSpace"   # dataset repo
RANGES_FILE    = "feature_ranges.json"

 
# Decision threshold (recall-biased: a missed fault costs more than a false alarm)
THRESHOLD = 0.45
 
# Feature order the model was trained on
FEATURES = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp",
]
 
# Fallback ranges (used only if feature_ranges.json cannot be downloaded).
# Derived from TRAIN-only stats during data preparation.
FALLBACK_RANGES = {
    "Engine rpm":       {"physical_min": 0.0,  "physical_max": 3000.0, "operating_min": 382.0,  "operating_max": 1565.0},
    "Lub oil pressure": {"physical_min": 0.1,  "physical_max": 10.0,   "operating_min": 0.86,   "operating_max": 5.61},
    "Fuel pressure":    {"physical_min": 0.1,  "physical_max": 25.0,   "operating_min": 1.40,   "operating_max": 16.16},
    "Coolant pressure": {"physical_min": 0.1,  "physical_max": 10.0,   "operating_min": 0.72,   "operating_max": 5.95},
    "lub oil temp":     {"physical_min": 40.0, "physical_max": 130.0,  "operating_min": 73.41,  "operating_max": 87.35},
    "Coolant temp":     {"physical_min": 40.0, "physical_max": 120.0,  "operating_min": 65.74,  "operating_max": 91.78},
}
 
# Sensible default (median-ish) starting positions + units for each slider
DEFAULTS = {
    "Engine rpm":       (746.0, "RPM",  382.0),
    "Lub oil pressure": (3.16,  "kPa",  0.86),
    "Fuel pressure":    (6.20,  "kPa",  1.41),
    "Coolant pressure": (2.17,  "kPa",  0.72),
    "lub oil temp":     (76.82, "deg C",   73.41),
    "Coolant temp":     (78.35, "deg C",   65.74),
}

# UI-only display names (the model keys above must stay exactly as trained)
DISPLAY_LABELS = {
    "Engine rpm":       "Engine",
    "Lub oil pressure": "Lubricant Oil Pressure",
    "Fuel pressure":    "Fuel Pressure",
    "Coolant pressure": "Coolant Pressure",
    "lub oil temp":     "Lubricant Oil Temperature",
    "Coolant temp":     "Coolant Temperature",
}


# ---------------------------------------------------------------------
# Cached resources: model + validation ranges
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Download and unpickle the trained pipeline from the HF model hub.

    """
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILENAME)
    return joblib.load(model_path)



@st.cache_data
def load_ranges():
    """Load the train-derived feature ranges; fall back to bundled constants."""
    try:
        path = hf_hub_download(
            repo_id=RANGES_REPO, filename=RANGES_FILE, repo_type="dataset"
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return FALLBACK_RANGES
 
 
def validate_reading(reading: dict, ranges: dict) -> dict:
    """Two-tier check: Tier-1 physical (block) + Tier-2 operating (warn)."""
    errors, warnings = [], []
    for feat, val in reading.items():
        b = ranges.get(feat, {})
        if "physical_min" in b and (val < b["physical_min"] or val > b["physical_max"]):
            errors.append(
                f"{feat} = {val:g} is outside the physical range "
                f"[{b['physical_min']:g}, {b['physical_max']:g}] — likely a sensor fault."
            )
        elif "operating_min" in b and (val < b["operating_min"] or val > b["operating_max"]):
            warnings.append(
                f"{feat} = {val:g} is outside the normal operating range "
                f"[{b['operating_min']:.2f}, {b['operating_max']:.2f}] — prediction confidence reduced."
            )
    status = "SENSOR_FAULT" if errors else ("OUT_OF_RANGE" if warnings else "OK")
    return {"ok": not errors, "status": status, "errors": errors, "warnings": warnings}
 
 
# ---------------------------------------------------------------------
# Load resources with graceful error handling
# ---------------------------------------------------------------------
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)
 
ranges = load_ranges()
 
 
# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("Engine Maintenance Predictor")
st.markdown(
    "Predict whether an engine is **Active (healthy)** or **Faulty (needs maintenance)** "
    "from six real-time sensor readings. Adjust the sliders to match the current sensor "
    "values and click **Run diagnostics**."
)
st.divider()
 
if not model_loaded:
    st.error(
        "The prediction model could not be loaded.\n\n"
        f"Details: `{load_error}`\n\n"
    )
    st.stop()
 
 
# ---------------------------------------------------------------------
# Sidebar: reference envelope
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Normal operating ranges")
    st.caption("Learned from training data (1st to 99th percentile). Readings outside these are flagged.")
    for feat in FEATURES:
        b = ranges[feat]
        unit = DEFAULTS[feat][1]
        st.markdown(f"**{DISPLAY_LABELS[feat]}**  \n{b['operating_min']:.2f} – {b['operating_max']:.2f} {unit}")
    st.divider()
    st.caption(f"Decision threshold: fault if P(faulty) ≥ {THRESHOLD:.0%}")
 
 
# ---------------------------------------------------------------------
# Sensor input sliders
# ---------------------------------------------------------------------
st.subheader("Sensor readings")
 
def sensor_slider(feat):
    b = ranges[feat]
    default, unit, step = DEFAULTS[feat]
    return st.slider(
        f"{DISPLAY_LABELS[feat]} ({unit})",
        min_value=float(b["physical_min"]),
        max_value=float(b["physical_max"]),
        value=float(default),
        step=float(step),
        help=f"Normal operating range: {b['operating_min']:.2f}–{b['operating_max']:.2f} {unit}",
    )
 
col1, col2 = st.columns(2)
with col1:
    engine_rpm      = sensor_slider("Engine rpm")
    lub_oil_press   = sensor_slider("Lub oil pressure")
    fuel_press      = sensor_slider("Fuel pressure")
with col2:
    coolant_press   = sensor_slider("Coolant pressure")
    lub_oil_temp    = sensor_slider("lub oil temp")
    coolant_temp    = sensor_slider("Coolant temp")
 
st.divider()
 
# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
if st.button("Run diagnostics", type="primary", use_container_width=True):
 
    reading = {
        "Engine rpm":       float(engine_rpm),
        "Lub oil pressure": float(lub_oil_press),
        "Fuel pressure":    float(fuel_press),
        "Coolant pressure": float(coolant_press),
        "lub oil temp":     float(lub_oil_temp),
        "Coolant temp":     float(coolant_temp),
    }
 
    # Two-tier input validation BEFORE the model is called
    check = validate_reading(reading, ranges)
 
    if not check["ok"]:
        st.error("**Sensor fault detected — prediction blocked.**")
        for e in check["errors"]:
            st.write(f"- {e}")
        st.info(
            "An out-of-physical-range reading is itself a maintenance signal. "
            "Inspect the flagged sensor(s) before trusting any prediction."
        )
        st.stop()
 
    # Predict
    X = pd.DataFrame([reading])[FEATURES]
    proba_faulty = float(model.predict_proba(X)[0, 1])
    is_faulty = proba_faulty >= THRESHOLD
 
    st.subheader("Diagnostic result")
    r1, r2 = st.columns([2, 1])
 
    with r1:
        if is_faulty:
            st.error(
                f"### Faulty — maintenance recommended\n"
                f"Estimated **{proba_faulty:.1%}** probability the engine requires maintenance."
            )
        else:
            st.success(
                f"### Active — operating normally\n"
                f"Estimated **{proba_faulty:.1%}** probability of a fault "
                f"(below the {THRESHOLD:.0%} alert threshold)."
            )
 
        # surface any Tier-2 warnings alongside the prediction
        if check["warnings"]:
            st.warning("**Reading outside normal operating range — treat with lower confidence:**")
            for w in check["warnings"]:
                st.write(f"- {w}")
 
    with r2:
        st.metric(
            label="P(faulty)",
            value=f"{proba_faulty:.1%}",
            delta=f"{proba_faulty - THRESHOLD:+.1%} vs threshold",
            delta_color="inverse",
        )
        st.progress(min(proba_faulty, 1.0))
 
    st.divider()
    st.subheader("Recommended action")
    if is_faulty:
        st.markdown(
            """
            **Schedule maintenance and inspect the engine.**
            - Prioritise inspection of the sensors reading furthest from their normal range
            - Check lubrication and cooling subsystems before returning the unit to service
            - Log this reading for the maintenance history / retraining dataset
            """
        )
    else:
        st.markdown(
            """
            **No immediate action required.**
            - Continue routine monitoring at the normal cadence
            - Re-run diagnostics if any sensor drifts toward its range boundary
            """
        )
 
    with st.expander("View the exact reading sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)
 
# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.divider()
st.caption(
    "Predictive maintenance demo • model served from the Hugging Face model hub • "
    "inputs validated against the training-derived operating envelope. "
    "Predictions are decision support, not a substitute for physical inspection."
)
