import math
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# CoolProp import
# -------------------------
try:
    from CoolProp.CoolProp import PropsSI
    COOLPROP_OK = True
except Exception:
    COOLPROP_OK = False

# =========================
# Simple psychrometrics (engineering-grade for evaporative equipment sizing)
# =========================
def p_ws_kpa(T_c: float) -> float:
    """Saturation vapor pressure over water (kPa), good for 0–60°C."""
    return 0.61094 * math.exp((17.625 * T_c) / (T_c + 243.04))

def humidity_ratio_from_tdb_twb(T_db: float, T_wb: float, P_kpa: float = 101.325) -> float:
    """Humidity ratio w (kg/kg_da) from dry bulb and wet bulb."""
    if T_wb > T_db:
        T_wb = T_db
    pws_wb = p_ws_kpa(T_wb)
    A = 0.00066 * (1.0 + 0.00115 * T_wb)
    p_w = pws_wb - A * P_kpa * (T_db - T_wb)
    p_w = max(0.0001, min(p_w, 0.98 * P_kpa))
    w = 0.621945 * p_w / (P_kpa - p_w)
    return max(0.0, w)

def enthalpy_moist_air_kj_per_kgda(T_db: float, w: float) -> float:
    """Moist air enthalpy (kJ/kg dry air)."""
    return 1.006 * T_db + w * (2501.0 + 1.86 * T_db)

def sat_air_enthalpy_at_T_kj_per_kgda(T_c: float, P_kpa: float = 101.325) -> float:
    """Enthalpy of saturated air at temperature T (kJ/kg_da)."""
    pws = p_ws_kpa(T_c)
    pws = min(pws, 0.98 * P_kpa)
    w_s = 0.621945 * pws / (P_kpa - pws)
    return enthalpy_moist_air_kj_per_kgda(T_c, w_s)

def air_density_kg_per_m3(T_db: float, w: float, P_kpa: float = 101.325) -> float:
    """Approx moist air density (kg/m³)."""
    P = P_kpa * 1000.0
    T_k = T_db + 273.15
    R_da = 287.055
    R_wv = 461.495
    p_w = P * w / (0.621945 + w)
    p_w = min(p_w, 0.98 * P)
    p_da = P - p_w
    rho = p_da / (R_da * T_k) + p_w / (R_wv * T_k)
    return rho

# =========================
# Fluid properties via CoolProp (Water / MEG / MPG)
# =========================
def make_process_fluid(fluid_choice: str, glycol_pct: int) -> str:
    """
    CoolProp fluid string:
      Water -> "Water"
      MEG 30% -> "INCOMP::MEG-30%"
      MPG 30% -> "INCOMP::MPG-30%"
    """
    if fluid_choice == "Water" or glycol_pct <= 0:
        return "Water"
    glycol_pct = int(glycol_pct)
    if fluid_choice == "MEG":
        return f"INCOMP::MEG-{glycol_pct}%"
    if fluid_choice == "MPG":
        return f"INCOMP::MPG-{glycol_pct}%"
    return "Water"

def fluid_props(fluid: str, T_c: float, P_kpa: float = 101.325):
    """
    Returns cp (kJ/kg-K), rho (kg/m3), mu (Pa.s), k (W/m-K).
    """
    if not COOLPROP_OK:
        raise RuntimeError("CoolProp is not available. Install CoolProp==6.6.0 in requirements.txt")

    T_k = T_c + 273.15
    P_pa = P_kpa * 1000.0

    cp = PropsSI("C", "T", T_k, "P", P_pa, fluid) / 1000.0   # kJ/kg-K
    rho = PropsSI("D", "T", T_k, "P", P_pa, fluid)          # kg/m3
    mu = PropsSI("V", "T", T_k, "P", P_pa, fluid)           # Pa.s
    k = PropsSI("L", "T", T_k, "P", P_pa, fluid)            # W/m-K
    return cp, rho, mu, k

# =========================
# Heat/mass transfer model (Merkel-style marching)
# dQ = K * dA * (hs(Tw) - ha)
# =========================
def merkel_required_area(
    Q_kw: float,
    Tw_in: float,
    Tw_out_target: float,
    Tdb_in: float,
    Twb_in: float,
    Vdot_air_m3_h: float,
    K_kg_s_m2: float,
    cp_kj_kgK: float,
    P_kpa: float = 101.325,
    dA_step_m2: float = 0.10,
    max_area_m2: float = 2000.0
):
    """
    Returns required wetted area (m2), and marching profile dataframe.
    """
    if Tw_out_target >= Tw_in:
        raise ValueError("Leaving water temperature must be lower than entering water temperature.")

    w_in = humidity_ratio_from_tdb_twb(Tdb_in, Twb_in, P_kpa)
    h_a = enthalpy_moist_air_kj_per_kgda(Tdb_in, w_in)
    rho_air = air_density_kg_per_m3(Tdb_in, w_in, P_kpa)

    Vdot_air_m3_s = Vdot_air_m3_h / 3600.0
    m_air = rho_air * Vdot_air_m3_s  # kg/s (moist air, good approx)

    dT = Tw_in - Tw_out_target
    m_w = Q_kw / (cp_kj_kgK * dT)  # kg/s because Q=kJ/s, cp=kJ/kgK

    A = 0.0
    Tw = Tw_in

    rows = []
    step = 0
    while Tw > Tw_out_target and A < max_area_m2:
        hs = sat_air_enthalpy_at_T_kj_per_kgda(Tw, P_kpa)
        drive = max(0.05, hs - h_a)  # avoid zero/negative drive

        dQ = K_kg_s_m2 * dA_step_m2 * drive  # kW
        # update
        h_a_new = h_a + dQ / max(1e-9, m_air)
        Tw_new = Tw - dQ / max(1e-9, (m_w * cp_kj_kgK))

        A += dA_step_m2
        rows.append([step, A, Tw, hs, h_a, drive, dQ, m_air, m_w])
        step += 1
        Tw, h_a = Tw_new, h_a_new

        # safety stop if numerics go odd
        if step > 300000:
            break

    df = pd.DataFrame(rows, columns=[
        "step", "Area_m2", "WaterTemp_C", "h_sat_kJkgda", "h_air_kJkgda", "Driving_h", "dQ_step_kW", "m_air_kg_s", "m_w_kg_s"
    ])
    return A, m_w, m_air, w_in, rho_air, df

# =========================
# Tube-side hydraulics & convection
# =========================
def reynolds(rho: float, v: float, D: float, mu: float) -> float:
    return rho * v * D / max(mu, 1e-12)

def prandtl(cp_kj_kgK: float, mu: float, k_w_mK: float) -> float:
    # cp in kJ/kgK -> J/kgK
    cp = cp_kj_kgK * 1000.0
    return cp * mu / max(k_w_mK, 1e-12)

def nusselt_dittus_boelter(Re: float, Pr: float, heating: bool = True) -> float:
    # turbulent internal flow; if cooling of fluid, exponent ~0.3; heating ~0.4
    n = 0.4 if heating else 0.3
    if Re < 3000:
        # laminar fallback (very rough): Nu ~ 3.66 (fully developed)
        return 3.66
    return 0.023 * (Re ** 0.8) * (Pr ** n)

def friction_factor(Re: float) -> float:
    if Re < 2000:
        return 64.0 / max(Re, 1e-12)
    # Blasius smooth pipe
    return 0.3164 / (Re ** 0.25)

def dp_darcy(rho: float, v: float, D: float, L: float, mu: float, K_minor: float = 3.0) -> float:
    Re = reynolds(rho, v, D, mu)
    f = friction_factor(Re)
    dp_f = f * (L / max(D, 1e-12)) * 0.5 * rho * v * v
    dp_m = K_minor * 0.5 * rho * v * v
    return dp_f + dp_m

# =========================
# Materials & tube options (for fabrication + wall resistance bookkeeping)
# =========================
TUBE_MATERIALS = {
    "Mild Steel (CS)": {"k_wall_W_mK": 45.0},
    "Stainless Steel 304": {"k_wall_W_mK": 16.0},
    "Stainless Steel 316": {"k_wall_W_mK": 14.0},
    "Mild Steel Hot Dip Galvanized (Zinc coated)": {"k_wall_W_mK": 45.0},
    "Copper": {"k_wall_W_mK": 385.0},
    "Cu-Ni 90/10": {"k_wall_W_mK": 50.0},
}

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Evaporative Cooler Coil Designer", layout="wide")
st.title("Forced-Draft Evaporative Fluid Cooler — Coil + Fan + Hydraulics Sizing (with Glycol & CoolProp)")

if not COOLPROP_OK:
    st.error("CoolProp is not installed in this environment. Add CoolProp==6.6.0 to requirements.txt.")
    st.stop()

st.caption(
    "This tool sizes required wetted coil area using a Merkel-style enthalpy marching model (calibratable K). "
    "Then it checks your coil geometry (tube OD/thickness/pitch/rows/circuits/headers) and estimates tube ΔP and fan power."
)

tab1, tab2, tab3 = st.tabs(["Inputs", "Results", "Calibration Notes"])

# -------------------------
# Inputs
# -------------------------
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Duty & Process Fluid")
        unitQ = st.radio("Heat rejection unit", ["kW", "kcal/h"], horizontal=True)
        Q_in = st.number_input("Heat rejection", min_value=1.0, value=105.0, step=1.0)

        Q_kw = Q_in if unitQ == "kW" else (Q_in * 1.163 / 1000.0)

        Tw_in = st.number_input("Process fluid inlet (hot) temp, °C", value=39.0, step=0.5)
        Tw_out = st.number_input("Process fluid outlet (cooled) temp, °C", value=33.0, step=0.5)

        fluid_choice = st.radio("Process fluid", ["Water", "MEG", "MPG"], horizontal=True)
        glycol_pct = st.slider("Glycol concentration (%)", min_value=0, max_value=60, value=0, step=5)

        flow_mode = st.radio("Flow input mode", ["Auto (from Q and ΔT)", "User flow (m³/h)"], horizontal=True)
        user_flow = st.number_input("User flow (m³/h)", value=15.0, step=0.5, disabled=(flow_mode != "User flow (m³/h)"))

    with col2:
        st.subheader("Ambient Air (Design)")
        P_kpa = st.number_input("Barometric pressure, kPa", value=101.325, step=0.1)
        Tdb = st.number_input("Entering air Dry Bulb, °C", value=42.0, step=0.5)
        Twb = st.number_input("Entering air Wet Bulb, °C", value=30.0, step=0.5)

        air_mode = st.radio("Airflow mode", ["User airflow (m³/h)", "Estimate from Δh (kJ/kg_da)"], horizontal=True)
        if air_mode == "User airflow (m³/h)":
            Vdot_air = st.number_input("Airflow through unit, m³/h", min_value=1000.0, value=22000.0, step=500.0)
            dh_assumed = st.number_input("Δh assumption (disabled)", value=15.0, step=1.0, disabled=True)
        else:
            dh_assumed = st.number_input("Assumed air enthalpy rise Δh, kJ/kg_da", min_value=5.0, value=15.0, step=1.0)
            Vdot_air = None

        st.subheader("Fan")
        dP_fan = st.number_input("Fan total static pressure, Pa", min_value=50.0, value=200.0, step=10.0)
        eta_fan = st.number_input("Fan+motor+drive efficiency (0–1)", min_value=0.2, max_value=0.85, value=0.60, step=0.05)

    with col3:
        st.subheader("Merkel Transfer (Calibratable)")
        K = st.number_input("K coefficient (kg/s·m²)", min_value=0.0001, max_value=0.01, value=0.0015, step=0.0001, format="%.4f")
        dA_step = st.number_input("Marching dA step (m²)", min_value=0.01, value=0.10, step=0.01)
        max_area = st.number_input("Max area limit (m²) safety", min_value=50.0, value=2000.0, step=50.0)

        st.subheader("Coil Geometry Inputs")
        tube_mat = st.selectbox("Tube material", list(TUBE_MATERIALS.keys()), index=0)
        Do_mm = st.number_input("Tube OD (mm)", min_value=6.0, value=25.4, step=0.5)
        t_mm = st.number_input("Tube thickness (mm)", min_value=0.5, value=2.5, step=0.1)

        rows_depth = st.number_input("Number of rows (depth, airflow direction)", min_value=1, value=6, step=1)

        vert_pitch_mm = st.number_input("Vertical pitch (mm)", min_value=15.0, value=50.0, step=1.0)
        horiz_pitch_mm = st.number_input("Horizontal pitch (mm)", min_value=15.0, value=50.0, step=1.0)

        face_W_m = st.number_input("Coil face width (m)", min_value=0.3, value=1.5, step=0.1)
        face_H_m = st.number_input("Coil face height (m)", min_value=0.3, value=1.5, step=0.1)

        tube_length_m = st.number_input("Tube straight length (m)", min_value=0.3, value=1.5, step=0.1)

        circuits = st.number_input("Number of parallel circuits", min_value=1, value=8, step=1)

        hdr_in_mm = st.number_input("Inlet header diameter (mm)", min_value=25.0, value=80.0, step=5.0)
        hdr_out_mm = st.number_input("Outlet header diameter (mm)", min_value=25.0, value=80.0, step=5.0)

        K_minor = st.number_input("Minor-loss coefficient per circuit (bends+entry/exit)", min_value=0.0, value=3.0, step=0.5)

run = st.button("Run Design & Check", type="primary")

# -------------------------
# Results calculation
# -------------------------
with tab2:
    st.subheader("Results")

    if not run:
        st.info("Enter inputs in the Inputs tab and click **Run Design & Check**.")
    else:
        # Fluid properties at mean temp
        T_mean = 0.5 * (Tw_in + Tw_out)
        proc_fluid = make_process_fluid(fluid_choice, glycol_pct)
        cp, rho, mu, k_fluid = fluid_props(proc_fluid, T_mean, P_kpa)

        # Air properties
        w_in = humidity_ratio_from_tdb_twb(Tdb, Twb, P_kpa)
        h_in = enthalpy_moist_air_kj_per_kgda(Tdb, w_in)
        rho_air = air_density_kg_per_m3(Tdb, w_in, P_kpa)

        # Estimate airflow if needed
        if air_mode == "Estimate from Δh (kJ/kg_da)":
            m_air_est = Q_kw / dh_assumed  # kg/s dry-air approx
            Vdot_air = (m_air_est / rho_air) * 3600.0

        # Merkel required area
        A_req, m_w_auto, m_air, w_in_calc, rho_air_calc, df_profile = merkel_required_area(
            Q_kw=Q_kw,
            Tw_in=Tw_in,
            Tw_out_target=Tw_out,
            Tdb_in=Tdb,
            Twb_in=Twb,
            Vdot_air_m3_h=Vdot_air,
            K_kg_s_m2=K,
            cp_kj_kgK=cp,
            P_kpa=P_kpa,
            dA_step_m2=dA_step,
            max_area_m2=max_area
        )

        # Determine flow
        if flow_mode == "User flow (m³/h)":
            flow_m3_h = user_flow
            m_w = (flow_m3_h * rho) / 3600.0  # kg/s
            Q_check = m_w * cp * (Tw_in - Tw_out)  # kW
        else:
            m_w = m_w_auto
            flow_m3_h = (m_w * 3600.0) / max(rho, 1e-9)
            Q_check = Q_kw

        # Coil tube count from pitch & face
        # Tubes across width and height (integer count)
        tubes_per_row = max(1, int(math.floor((face_W_m * 1000.0) / horiz_pitch_mm)))
        tubes_in_height = max(1, int(math.floor((face_H_m * 1000.0) / vert_pitch_mm)))

        # Total tubes per "row layer" = tubes_per_row * tubes_in_height
        tubes_per_layer = tubes_per_row * tubes_in_height

        # Total straight tube pieces = tubes_per_layer * rows_depth
        total_tubes = tubes_per_layer * rows_depth

        # Total tube length (straight only)
        L_total = total_tubes * tube_length_m

        # External area provided
        Do_m = Do_mm / 1000.0
        A_provided = math.pi * Do_m * L_total

        # Tube internal diameter
        Di_mm = max(0.5, Do_mm - 2.0 * t_mm)
        Di_m = Di_mm / 1000.0

        # Circuit length approximation:
        # In a real serpentine coil, each circuit passes through multiple rows; for first estimate:
        L_circuit = L_total / max(circuits, 1)

        # Internal velocity per circuit
        flow_m3_s = flow_m3_h / 3600.0
        flow_per_circuit = flow_m3_s / max(circuits, 1)
        A_id = math.pi * (Di_m ** 2) / 4.0
        v_int = flow_per_circuit / max(A_id, 1e-12)

        # Reynolds, Prandtl, hi
        Re = reynolds(rho, v_int, Di_m, mu)
        Pr = prandtl(cp, mu, k_fluid)
        Nu = nusselt_dittus_boelter(Re, Pr, heating=False)
        h_i = Nu * k_fluid / max(Di_m, 1e-12)  # W/m2-K

        # Tube dp per circuit (Pa)
        dp_pa = dp_darcy(rho, v_int, Di_m, L_circuit, mu, K_minor=K_minor)

        # Header velocities (sanity)
        hdr_in_m = hdr_in_mm / 1000.0
        hdr_out_m = hdr_out_mm / 1000.0
        A_hdr_in = math.pi * hdr_in_m**2 / 4.0
        A_hdr_out = math.pi * hdr_out_m**2 / 4.0
        v_hdr_in = flow_m3_s / max(A_hdr_in, 1e-12)
        v_hdr_out = flow_m3_s / max(A_hdr_out, 1e-12)

        # Fan power
        Vdot_air_m3_s = Vdot_air / 3600.0
        fan_kw = (Vdot_air_m3_s * dP_fan) / max(eta_fan, 1e-9) / 1000.0

        # Pass/fail
        margin_pct = (A_provided / max(A_req, 1e-9) - 1.0) * 100.0

        # --- Display key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duty (kW)", f"{Q_kw:.1f}")
        c2.metric("Required wetted area (m²)", f"{A_req:.1f}")
        c3.metric("Provided coil area (m²)", f"{A_provided:.1f}")
        c4.metric("Area margin (%)", f"{margin_pct:+.1f}")

        st.write("### Process Fluid Properties (CoolProp @ mean temperature)")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Fluid", proc_fluid)
        c6.metric("cp (kJ/kg·K)", f"{cp:.3f}")
        c7.metric("ρ (kg/m³)", f"{rho:.1f}")
        c8.metric("μ (mPa·s)", f"{mu*1000.0:.3f}")

        st.write("### Flow & Coil Counts")
        c9, c10, c11, c12 = st.columns(4)
        c9.metric("Process flow (m³/h)", f"{flow_m3_h:.2f}")
        c10.metric("Tubes across width", f"{tubes_per_row:d}")
        c11.metric("Tubes in height", f"{tubes_in_height:d}")
        c12.metric("Total tubes (straight)", f"{total_tubes:d}")

        st.caption("Tube counts are computed as floor(face dimension / pitch). Adjust face sizes/pitches to hit your desired count.")

        st.write("### Tube-side Hydraulics (per circuit, first estimate)")
        c13, c14, c15, c16 = st.columns(4)
        c13.metric("Tube ID (mm)", f"{Di_mm:.2f}")
        c14.metric("Internal velocity (m/s)", f"{v_int:.2f}")
        c15.metric("Re", f"{Re:.0f}")
        c16.metric("ΔP per circuit (kPa)", f"{dp_pa/1000.0:.1f}")

        st.write("### Tube-side Heat Transfer (internal)")
        c17, c18, c19 = st.columns(3)
        c17.metric("Pr", f"{Pr:.2f}")
        c18.metric("Nu", f"{Nu:.1f}")
        c19.metric("hᵢ (W/m²·K)", f"{h_i:.0f}")

        st.write("### Header velocity sanity check")
        c20, c21 = st.columns(2)
        c20.metric("Inlet header velocity (m/s)", f"{v_hdr_in:.2f}")
        c21.metric("Outlet header velocity (m/s)", f"{v_hdr_out:.2f}")
        st.caption("Typical good practice: headers ~1–2.5 m/s. Too high = noise/erosion; too low = poor distribution risk.")

        st.write("### Air & Fan")
        c22, c23, c24, c25 = st.columns(4)
        c22.metric("Airflow (m³/h)", f"{Vdot_air:.0f}")
        c23.metric("Air density (kg/m³)", f"{rho_air_calc:.2f}")
        c24.metric("Fan static (Pa)", f"{dP_fan:.0f}")
        c25.metric("Fan shaft power (kW)", f"{fan_kw:.2f}")
        st.caption("Select motor with margin (typically 1.25–1.5×) and use VFD for stable leaving water temperature control.")

        # Provide a compact summary table for exporting
        summary = {
            "Q_kW": Q_kw,
            "Tw_in_C": Tw_in,
            "Tw_out_C": Tw_out,
            "Tdb_in_C": Tdb,
            "Twb_in_C": Twb,
            "Airflow_m3_h": Vdot_air,
            "K_kg_s_m2": K,
            "A_required_m2": A_req,
            "A_provided_m2": A_provided,
            "Area_margin_pct": margin_pct,
            "Process_fluid": proc_fluid,
            "cp_kJ_kgK": cp,
            "rho_kg_m3": rho,
            "mu_mPa_s": mu * 1000.0,
            "k_W_mK": k_fluid,
            "Flow_m3_h": flow_m3_h,
            "Tube_OD_mm": Do_mm,
            "Tube_thickness_mm": t_mm,
            "Tube_ID_mm": Di_mm,
            "Rows_depth": rows_depth,
            "Vert_pitch_mm": vert_pitch_mm,
            "Horiz_pitch_mm": horiz_pitch_mm,
            "Face_W_m": face_W_m,
            "Face_H_m": face_H_m,
            "Tubes_per_row": tubes_per_row,
            "Tubes_in_height": tubes_in_height,
            "Total_tubes": total_tubes,
            "Tube_straight_length_m": tube_length_m,
            "Total_tube_length_m": L_total,
            "Circuits": circuits,
            "Circuit_length_m_est": L_circuit,
            "Tube_velocity_m_s": v_int,
            "Re": Re,
            "DP_circuit_kPa": dp_pa/1000.0,
            "Header_in_mm": hdr_in_mm,
            "Header_out_mm": hdr_out_mm,
            "Header_v_in_m_s": v_hdr_in,
            "Header_v_out_m_s": v_hdr_out,
            "Fan_static_Pa": dP_fan,
            "Fan_eff": eta_fan,
            "Fan_power_kW": fan_kw,
        }
        df_sum = pd.DataFrame([summary])
        st.write("### Export")
        st.download_button(
            "Download summary CSV",
            data=df_sum.to_csv(index=False).encode("utf-8"),
            file_name="evap_fluid_cooler_summary.csv",
            mime="text/csv"
        )

        with st.expander("Show Merkel marching profile (last 50 rows)"):
            st.dataframe(df_profile.tail(50), use_container_width=True)

# -------------------------
# Calibration Notes
# -------------------------
with tab3:
    st.markdown(
        """
### What is **K** and why it exists?

The Merkel model couples evaporation + convection using an enthalpy driving force.  
In real equipment, performance depends on spray distribution, wetting, airflow maldistribution, drift eliminator losses, etc.

So we use a **calibratable** coefficient **K (kg/s·m²)**.  
This is normal OEM practice: you tune K to one known reference unit / test, then reuse it for your standard geometry family.

### How to calibrate K (recommended)
1. Take any known unit (even competitor):
   - known duty (kW)
   - known WB, water in/out, airflow (or fan size)
   - known coil geometry (tube length, OD → coil area)
2. Run the model and adjust **K** until:
   - `Required area ≈ Provided area` for that unit.
3. Lock K for that coil family.

### Contract tip
Always quote performance at:
- design WB (Dubai 30°C)
- specified water in/out and flow
- “clean coil + proper water treatment” condition
        """
    )
