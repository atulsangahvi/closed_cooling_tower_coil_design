import math
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Psychrometrics (simple, robust enough for engineering sizing)
# =========================

def p_ws_kpa(T_c: float) -> float:
    """
    Saturation vapor pressure over water (kPa).
    Magnus-Tetens style; good for 0-60C range.
    """
    return 0.61094 * math.exp((17.625 * T_c) / (T_c + 243.04))

def humidity_ratio_from_tdb_twb(T_db: float, T_wb: float, P_kpa: float = 101.325) -> float:
    """
    Humidity ratio w (kg/kg_da) from dry-bulb and wet-bulb, using a ventilated psychrometer relation.
    Engineering approximation (ASHRAE-style).
    """
    if T_wb > T_db:
        T_wb = T_db

    pws_wb = p_ws_kpa(T_wb)
    # psychrometric coefficient (per ASHRAE approximation)
    A = 0.00066 * (1.0 + 0.00115 * T_wb)  # 1/°C
    # partial pressure of water vapor
    p_w = pws_wb - A * P_kpa * (T_db - T_wb)
    p_w = max(0.0001, min(p_w, 0.98 * P_kpa))
    w = 0.621945 * p_w / (P_kpa - p_w)
    return max(0.0, w)

def enthalpy_moist_air_kj_per_kgda(T_db: float, w: float) -> float:
    """
    Moist air enthalpy (kJ/kg dry air).
    h = 1.006*T + w*(2501 + 1.86*T)
    """
    return 1.006 * T_db + w * (2501.0 + 1.86 * T_db)

def sat_air_enthalpy_at_T_kj_per_kgda(T_c: float, P_kpa: float = 101.325) -> float:
    """
    Enthalpy of saturated air at temperature T (kJ/kg_da).
    """
    pws = p_ws_kpa(T_c)
    pws = min(pws, 0.98 * P_kpa)
    w_s = 0.621945 * pws / (P_kpa - pws)
    return enthalpy_moist_air_kj_per_kgda(T_c, w_s)

def air_density_kg_per_m3(T_db: float, w: float, P_kpa: float = 101.325) -> float:
    """
    Approx moist air density (kg/m3). Good enough for fan power sizing.
    """
    # Convert to Pa
    P = P_kpa * 1000.0
    T_k = T_db + 273.15
    # gas constants
    R_da = 287.055  # J/kg-K
    R_wv = 461.495  # J/kg-K
    # partial pressures
    p_ws = p_ws_kpa(T_db) * 1000.0
    # estimate vapor pressure from w:
    # w = 0.621945*p_w/(P - p_w) => p_w = P*w/(0.621945 + w)
    p_w = P * w / (0.621945 + w)
    p_w = min(p_w, 0.98 * P)
    p_da = P - p_w
    rho = p_da / (R_da * T_k) + p_w / (R_wv * T_k)
    return rho

# =========================
# Merkel-style marching model
# =========================

def merkel_required_area(
    Q_kw: float,
    Tw_in: float,
    Tw_out_target: float,
    Tdb_in: float,
    Twb_in: float,
    Vdot_air_m3_h: float,
    K_kg_s_m2: float,
    P_kpa: float = 101.325,
    n_seg: int = 50
):
    """
    Merkel-style integration:
    dQ = K * dA * (h_s(Tw) - h_a)
    dh_a = dQ / m_air
    dTw = -dQ / (m_w * cp)

    We integrate over area until Tw reaches Tw_out_target.
    Returns: required area (m2), plus a profile dataframe.
    """
    if Tw_out_target >= Tw_in:
        raise ValueError("Leaving water temperature must be lower than entering water temperature.")

    # Air entering properties
    w_in = humidity_ratio_from_tdb_twb(Tdb_in, Twb_in, P_kpa)
    h_a = enthalpy_moist_air_kj_per_kgda(Tdb_in, w_in)  # kJ/kg_da

    rho_air = air_density_kg_per_m3(Tdb_in, w_in, P_kpa)
    Vdot_air_m3_s = Vdot_air_m3_h / 3600.0
    m_air = rho_air * Vdot_air_m3_s  # kg/s moist air (close to kg/s dry air for engineering)

    # Process water mass flow derived from Q and ΔT (energy balance)
    cp_w = 4.186  # kJ/kg-K
    dT = Tw_in - Tw_out_target
    m_w = (Q_kw) / (cp_w * dT)  # kg/s because Q in kJ/s

    # integrate in small area steps; we don't know A, so we adapt with dA
    # We'll use a fixed dA guess and iterate until hit target Tw_out.
    # Choose small dA to ensure numerical stability:
    dA = 0.1  # m2 per step
    A = 0.0
    Tw = Tw_in

    rows = []
    max_steps = int(200000)  # safety

    for i in range(max_steps):
        hs = sat_air_enthalpy_at_T_kj_per_kgda(Tw, P_kpa)  # kJ/kg_da
        dh_drive = max(0.1, hs - h_a)  # avoid zero driving force

        dQ = K_kg_s_m2 * dA * dh_drive  # (kg/s/m2)*(m2)*(kJ/kg) = kJ/s = kW

        # Update states
        h_a_new = h_a + dQ / max(1e-6, m_air)
        Tw_new = Tw - dQ / max(1e-6, (m_w * cp_w))

        A += dA
        rows.append([A, Tw, hs, h_a, dh_drive, dQ])

        Tw, h_a = Tw_new, h_a_new

        if Tw <= Tw_out_target:
            break

        # if somehow water temp drops below wet-bulb by too much, still okay;
        # but if driving force collapses, K may be too low or airflow too low
        if A > 2000:  # sanity check for absurd sizes
            break

    df = pd.DataFrame(rows, columns=[
        "Area_m2", "WaterTemp_C", "h_sat_kJkgda", "h_air_kJkgda", "Driving_(hs-ha)", "dQ_step_kW"
    ])

    return A, m_w, m_air, w_in, rho_air, df

# =========================
# Geometry helpers
# =========================

TUBE_CATALOG = {
    "1\" OD (25.4 mm)": {"Do_mm": 25.4, "ID_mm_default": 22.0},
    "7/8\" OD (22.2 mm)": {"Do_mm": 22.2, "ID_mm_default": 19.0},
    "1-1/4\" OD (31.8 mm)": {"Do_mm": 31.8, "ID_mm_default": 28.0},
    "1-1/2\" OD (38.1 mm)": {"Do_mm": 38.1, "ID_mm_default": 34.0},
}

MATERIALS = {
    "Carbon Steel": {"k_W_mK": 45.0},
    "HDG (Galvanized Steel)": {"k_W_mK": 45.0},
    "SS304": {"k_W_mK": 16.0},
    "SS316": {"k_W_mK": 14.0},
}

def tube_area_from_length(Do_mm: float, L_m: float) -> float:
    Do_m = Do_mm / 1000.0
    return math.pi * Do_m * L_m

def suggest_circuits(flow_m3_h: float, ID_mm: float, v_target_m_s: float) -> int:
    ID_m = ID_mm / 1000.0
    A_flow = math.pi * (ID_m**2) / 4.0
    flow_m3_s = flow_m3_h / 3600.0
    # flow per circuit at v_target
    flow_per = v_target_m_s * A_flow
    n = int(math.ceil(flow_m3_s / max(1e-9, flow_per)))
    return max(1, n)

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Evaporative Fluid Cooler Designer", layout="wide")
st.title("Forced-Draft Evaporative Fluid Cooler (Closed-Loop Coil) — Sizing App")

st.caption(
    "This app uses a Merkel-style marching model (enthalpy driving force) that can be calibrated via the K coefficient. "
    "Use it for quoting and preliminary manufacturing design. For certified rating, validate against test data."
)

tab1, tab2, tab3 = st.tabs(["1) Inputs", "2) Results", "3) Notes & Calibration"])

with tab1:
    colA, colB, colC = st.columns(3)

    with colA:
        st.subheader("Duty & Water")
        unitQ = st.radio("Heat rejection input unit", ["kW", "kcal/h"], horizontal=True)
        Q_in = st.number_input("Heat rejection", min_value=1.0, value=105.0, step=1.0)

        if unitQ == "kcal/h":
            Q_kw = Q_in * 1.163 / 1000.0  # kcal/h -> W -> kW
        else:
            Q_kw = Q_in

        Tw_in = st.number_input("Entering (hot) process water temp, °C", value=39.0, step=0.5)
        Tw_out = st.number_input("Leaving (cooled) process water temp, °C", value=33.0, step=0.5)

        flow_mode = st.radio(
            "Process water flow mode",
            ["Auto from Q and ΔT (recommended)", "I will enter process flow"],
            horizontal=False
        )
        flow_user = st.number_input("Process water flow (m³/h)", value=15.0, step=0.5, disabled=(flow_mode != "I will enter process flow"))

    with colB:
        st.subheader("Ambient Air (Design)")
        P_kpa = st.number_input("Barometric pressure, kPa", value=101.325, step=0.1)
        Tdb = st.number_input("Entering air Dry Bulb, °C", value=42.0, step=0.5)
        Twb = st.number_input("Entering air Wet Bulb, °C", value=30.0, step=0.5)

        air_mode = st.radio(
            "Airflow mode",
            ["I will enter airflow (recommended)", "Estimate airflow from Δh assumption"],
            horizontal=False
        )
        if air_mode == "I will enter airflow (recommended)":
            Vdot_air = st.number_input("Airflow through unit, m³/h", min_value=1000.0, value=22000.0, step=500.0)
            dh_assumed = st.number_input("Assumed air enthalpy rise Δh, kJ/kg_da", value=15.0, step=1.0, disabled=True)
        else:
            dh_assumed = st.number_input("Assumed air enthalpy rise Δh, kJ/kg_da", min_value=5.0, value=15.0, step=1.0)
            # airflow will be estimated later
            Vdot_air = None

    with colC:
        st.subheader("Transfer & Fan")
        st.write("**Merkel transfer coefficient K** (calibratable)")
        K_default = 0.0015
        K = st.number_input(
            "K (kg/s·m²), typical starting 0.001–0.003",
            min_value=0.0001, max_value=0.0100, value=K_default, step=0.0001, format="%.4f"
        )
        n_seg = st.slider("Integration resolution (segments)", min_value=20, max_value=200, value=60, step=10)

        st.write("**Fan**")
        dP = st.number_input("Fan total static pressure, Pa", min_value=50.0, value=200.0, step=10.0)
        eta = st.number_input("Fan+motor+drive efficiency (0–1)", min_value=0.2, max_value=0.85, value=0.60, step=0.05)

st.divider()

# Run button
run = st.button("Run sizing calculation", type="primary")

# Store results in session state
if run:
    try:
        # Estimate airflow if needed
        w_in = humidity_ratio_from_tdb_twb(Tdb, Twb, P_kpa)
        h_in = enthalpy_moist_air_kj_per_kgda(Tdb, w_in)
        rho = air_density_kg_per_m3(Tdb, w_in, P_kpa)

        if air_mode == "Estimate airflow from Δh assumption":
            # m_air = Q / Δh ; V = m / rho
            m_air_est = Q_kw / dh_assumed
            Vdot_air = (m_air_est / rho) * 3600.0

        # If user wants to override process flow, adjust Q check
        if flow_mode == "I will enter process flow":
            # Recompute implied Q from user flow and ΔT, compare
            cp_w = 4.186
            m_w_user = (flow_user * 1000.0) / 3600.0  # kg/s
            Q_implied = m_w_user * cp_w * (Tw_in - Tw_out)  # kW
            Q_used = Q_kw
            flow_used = flow_user
        else:
            Q_used = Q_kw
            flow_used = None

        A_req, m_w_auto, m_air, w_in_calc, rho_air, df_profile = merkel_required_area(
            Q_kw=Q_used,
            Tw_in=Tw_in,
            Tw_out_target=Tw_out,
            Tdb_in=Tdb,
            Twb_in=Twb,
            Vdot_air_m3_h=Vdot_air,
            K_kg_s_m2=K,
            P_kpa=P_kpa,
            n_seg=n_seg
        )

        # Determine process water flow (m3/h)
        if flow_mode == "I will enter process flow":
            m_w = (flow_user * 1000.0) / 3600.0
            flow_m3_h = flow_user
            Q_check = (m_w * 4.186 * (Tw_in - Tw_out))
        else:
            m_w = m_w_auto
            flow_m3_h = (m_w * 3600.0) / 1000.0
            Q_check = Q_used

        # Fan power
        Vdot_air_m3_s = Vdot_air / 3600.0
        P_fan_kw = (Vdot_air_m3_s * dP) / max(1e-9, eta) / 1000.0

        # Coil geometry suggestions
        tube_sel = st.session_state.get("tube_sel", "1\" OD (25.4 mm)")
        st.session_state["last_results"] = {
            "Q_kw": Q_used,
            "A_req_m2": A_req,
            "Vdot_air_m3_h": Vdot_air,
            "m_air_kg_s": m_air,
            "m_w_kg_s": m_w,
            "flow_m3_h": flow_m3_h,
            "w_in": w_in_calc,
            "rho_air": rho_air,
            "fan_power_kw": P_fan_kw,
            "df_profile": df_profile,
            "Q_check": Q_check,
        }

        st.success("Sizing complete. Open the Results tab.")
    except Exception as e:
        st.error(f"Error: {e}")

with tab2:
    st.subheader("Results")
    res = st.session_state.get("last_results", None)

    if not res:
        st.info("Run the calculation from the Inputs tab.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Heat Rejection (kW)", f"{res['Q_kw']:.1f}")
        c2.metric("Required Wetted Coil Area (m²)", f"{res['A_req_m2']:.1f}")
        c3.metric("Airflow (m³/h)", f"{res['Vdot_air_m3_h']:.0f}")
        c4.metric("Fan Shaft Power (kW)", f"{res['fan_power_kw']:.2f}")

        st.write("### Water & Air States")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Process Water Flow (m³/h)", f"{res['flow_m3_h']:.2f}")
        c6.metric("Air mass flow (kg/s)", f"{res['m_air_kg_s']:.2f}")
        c7.metric("Inlet humidity ratio w (kg/kg_da)", f"{res['w_in']:.4f}")
        c8.metric("Air density (kg/m³)", f"{res['rho_air']:.2f}")

        st.write("### Merkel Marching Profile (debug/insight)")
        st.dataframe(res["df_profile"].tail(30), use_container_width=True)

        # Coil geometry helper
        st.write("### Coil Geometry Helper (convert area → tube length & circuits)")
        g1, g2, g3 = st.columns(3)

        with g1:
            mat = st.selectbox("Tube material", list(MATERIALS.keys()), index=0)
            tube = st.selectbox("Tube size", list(TUBE_CATALOG.keys()), index=0)
            Do_mm = TUBE_CATALOG[tube]["Do_mm"]

        with g2:
            ID_mm = st.number_input("Tube ID (mm) for velocity calc", value=float(TUBE_CATALOG[tube]["ID_mm_default"]), step=0.5)
            v_target = st.number_input("Target internal velocity (m/s)", value=1.5, step=0.1)

        with g3:
            # Convert required area to tube length
            L_req = res["A_req_m2"] / (math.pi * (Do_mm / 1000.0))
            circuits = suggest_circuits(res["flow_m3_h"], ID_mm, v_target)
            st.metric("Tube length required (m)", f"{L_req:.0f}")
            st.metric("Suggested parallel circuits", f"{circuits:d}")

        st.caption(
            "Tube length is based on external area = π·Do·L. Actual build must consider rows, headers, bends, and manufacturability."
        )

with tab3:
    st.subheader("Calibration & How to Use Like a Real OEM")

    st.markdown(
        """
### What BAC / EVAPCO are doing “internally” (in plain language)

They use a Merkel-style model, but their key advantage is **calibrated coefficients** that map:
- coil geometry (rows, pitch, tube OD)
- spray rate and nozzle pattern
- airside pressure drop (coil + eliminators + louvers)
- drift eliminator efficiency
- fouling assumptions

In this app, that “secret sauce” is represented by **K (kg/s·m²)**.

### How you should calibrate K (recommended)
If you have **one reference unit** (even a competitor’s) with a known rating:
- Known: Q, WB, airflow (or fan size), water in/out
- Adjust K until the model reproduces the known coil area (or known coil length)

Then keep that K for your standard geometry family.
This makes your quoting consistent and defensible.

### Quick sanity ranges
- If your calculated coil area is *crazy large* → K too low or airflow too low.
- If your calculated coil area is *too small* → K too high (over-optimistic) or you assumed too much airflow.

### Contract language tip
Always state performance at:
- Design WB (30°C for Dubai)
- Clean condition and with proper water treatment
- Specify flow and temperatures explicitly
        """
    )
