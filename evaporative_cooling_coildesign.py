import math
import io
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# Schematic plotting
# -------------------------
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc

# -------------------------
# PDF generation (ReportLab)
# -------------------------
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image as RLImage, LongTable
)
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# -------------------------
# CoolProp import
# -------------------------
try:
    from CoolProp.CoolProp import PropsSI
    COOLPROP_OK = True
except Exception:
    COOLPROP_OK = False


# ============================================================
# Psychrometrics (engineering-grade approximations)
# ============================================================
def p_ws_kpa(T_c: float) -> float:
    """Saturation vapor pressure over water (kPa), good for ~0–60°C."""
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


# ============================================================
# Fluid properties via CoolProp (Water / MEG / MPG)
# ============================================================
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
    """Returns cp (kJ/kg-K), rho (kg/m3), mu (Pa.s), k (W/m-K)."""
    if not COOLPROP_OK:
        raise RuntimeError("CoolProp not available. Use CoolProp==6.7.0 for Streamlit Python 3.13.")
    T_k = T_c + 273.15
    P_pa = P_kpa * 1000.0
    cp = PropsSI("C", "T", T_k, "P", P_pa, fluid) / 1000.0
    rho = PropsSI("D", "T", T_k, "P", P_pa, fluid)
    mu = PropsSI("V", "T", T_k, "P", P_pa, fluid)
    k = PropsSI("L", "T", T_k, "P", P_pa, fluid)
    return cp, rho, mu, k


# ============================================================
# Merkel-style marching model
# dQ = K * dA * (hs(Tw) - ha)
# ============================================================
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
    if Tw_out_target >= Tw_in:
        raise ValueError("Leaving fluid temperature must be lower than entering temperature.")

    w_in = humidity_ratio_from_tdb_twb(Tdb_in, Twb_in, P_kpa)
    h_a = enthalpy_moist_air_kj_per_kgda(Tdb_in, w_in)
    rho_air = air_density_kg_per_m3(Tdb_in, w_in, P_kpa)

    Vdot_air_m3_s = Vdot_air_m3_h / 3600.0
    m_air = rho_air * Vdot_air_m3_s

    dT = Tw_in - Tw_out_target
    m_w = Q_kw / (cp_kj_kgK * dT)

    A = 0.0
    Tw = Tw_in
    rows = []
    step = 0

    while Tw > Tw_out_target and A < max_area_m2:
        hs = sat_air_enthalpy_at_T_kj_per_kgda(Tw, P_kpa)
        drive = max(0.05, hs - h_a)

        dQ = K_kg_s_m2 * dA_step_m2 * drive
        h_a_new = h_a + dQ / max(1e-9, m_air)
        Tw_new = Tw - dQ / max(1e-9, (m_w * cp_kj_kgK))

        A += dA_step_m2
        rows.append([step, A, Tw, hs, h_a, drive, dQ, m_air, m_w])
        step += 1
        Tw, h_a = Tw_new, h_a_new

        if step > 300000:
            break

    df = pd.DataFrame(rows, columns=[
        "step", "Area_m2", "WaterTemp_C", "h_sat_kJkgda", "h_air_kJkgda",
        "Driving_h", "dQ_step_kW", "m_air_kg_s", "m_w_kg_s"
    ])
    return A, m_w, m_air, w_in, rho_air, df


# ============================================================
# Tube-side hydraulics & convection (simple first-pass)
# ============================================================
def reynolds(rho: float, v: float, D: float, mu: float) -> float:
    return rho * v * D / max(mu, 1e-12)


def prandtl(cp_kj_kgK: float, mu: float, k_w_mK: float) -> float:
    cp = cp_kj_kgK * 1000.0
    return cp * mu / max(k_w_mK, 1e-12)


def nusselt_dittus_boelter(Re: float, Pr: float, heating: bool = True) -> float:
    n = 0.4 if heating else 0.3
    if Re < 3000:
        return 3.66
    return 0.023 * (Re ** 0.8) * (Pr ** n)


def friction_factor(Re: float) -> float:
    if Re < 2000:
        return 64.0 / max(Re, 1e-12)
    return 0.3164 / (Re ** 0.25)


def dp_darcy(rho: float, v: float, D: float, L: float, mu: float, K_minor: float = 3.0) -> float:
    Re = reynolds(rho, v, D, mu)
    f = friction_factor(Re)
    dp_f = f * (L / max(D, 1e-12)) * 0.5 * rho * v * v
    dp_m = K_minor * 0.5 * rho * v * v
    return dp_f + dp_m


# ============================================================
# Materials list (selection)
# ============================================================
TUBE_MATERIALS = {
    "Mild Steel (CS)": {"k_wall_W_mK": 45.0},
    "Stainless Steel 304": {"k_wall_W_mK": 16.0},
    "Stainless Steel 316": {"k_wall_W_mK": 14.0},
    "Mild Steel Hot Dip Galvanized (Zinc coated)": {"k_wall_W_mK": 45.0},
    "Copper": {"k_wall_W_mK": 385.0},
    "Cu-Ni 90/10": {"k_wall_W_mK": 50.0},
}


# ============================================================
# Schematics (3 views)
# 1) Plan view (WIDTH vs DEPTH rows)
# 2) Side view (DEPTH rows vs HEIGHT)
# 3) Serpentine (one circuit): headers + passes + U-bends, even/odd rule
# ============================================================
def draw_three_views(
    face_W_m: float,
    face_H_m: float,
    rows_depth: int,
    horiz_pitch_mm: float,
    vert_pitch_mm: float,
    Do_mm: float,
    n_passes_air: int,
    L_straight_m: float
):
    rows_depth = max(1, int(rows_depth))
    n_passes_air = max(2, int(n_passes_air))
    hp_m = max(1e-6, horiz_pitch_mm / 1000.0)
    vp_m = max(1e-6, vert_pitch_mm / 1000.0)

    tubes_across = max(1, int(math.floor(face_W_m / hp_m)))
    tubes_high = max(1, int(math.floor(face_H_m / vp_m)))

    r_draw = 0.18

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # --------------------
    # View 1: Plan view
    # --------------------
    for r in range(rows_depth):
        for c in range(tubes_across):
            ax1.add_patch(Circle((c, r), radius=r_draw, fill=False, linewidth=1.2))
    ax1.add_patch(Rectangle((-0.5, -0.5), (tubes_across - 1) + 1.0, (rows_depth - 1) + 1.0,
                            fill=False, linewidth=1.0, linestyle="--"))
    ax1.set_title("View 1: Plan\n(WIDTH vs DEPTH rows)")
    ax1.text(0.0, -1.2,
             f"W={face_W_m:.3f} m, H-pitch={horiz_pitch_mm:.1f} mm → tubes_across={tubes_across}",
             fontsize=9, ha="left")
    ax1.text(0.0, -1.55, f"Depth rows={rows_depth}", fontsize=9, ha="left")
    ax1.set_xlim(-0.8, max(1.0, tubes_across - 1 + 0.8))
    ax1.set_ylim(-2.0, max(1.0, rows_depth - 1 + 0.8))
    ax1.set_aspect("equal", adjustable="box")
    ax1.axis("off")

    # --------------------
    # View 2: Side view
    # --------------------
    for r in range(rows_depth):
        for k in range(tubes_high):
            ax2.add_patch(Circle((r, k), radius=r_draw, fill=False, linewidth=1.2))
    ax2.add_patch(Rectangle((-0.5, -0.5), (rows_depth - 1) + 1.0, (tubes_high - 1) + 1.0,
                            fill=False, linewidth=1.0, linestyle="--"))
    ax2.set_title("View 2: Side\n(DEPTH rows vs HEIGHT)")
    ax2.text(0.0, -1.2,
             f"H={face_H_m:.3f} m, V-pitch={vert_pitch_mm:.1f} mm → tubes_high={tubes_high}",
             fontsize=9, ha="left")
    ax2.text(0.0, -1.55, f"Tube OD={Do_mm:.1f} mm (schematic)", fontsize=9, ha="left")
    ax2.set_xlim(-0.8, max(1.0, rows_depth - 1 + 0.8))
    ax2.set_ylim(-2.0, max(1.0, tubes_high - 1 + 0.8))
    ax2.set_aspect("equal", adjustable="box")
    ax2.axis("off")

    # --------------------
    # View 3: One serpentine circuit
    # Coordinate system:
    #   x = 0 is header plane (where headers/nozzles are)
    #   x increases to the U-bend end.
    #   y is pass index (vertical)
    # Even passes -> headers on same side.
    # Odd passes  -> headers opposite sides.
    # --------------------
    L = max(0.2, float(L_straight_m))
    dy = 1.0  # vertical spacing in drawing units
    y_max = (n_passes_air - 1) * dy

    same_side = (n_passes_air % 2 == 0)
    hdr_out_side = "Same side" if same_side else "Opposite side"

    # headers (schematic)
    ax3.plot([0, 0], [-0.5, y_max + 0.5], linewidth=6, alpha=0.7)  # header plane

    # draw serpentine passes
    for i in range(n_passes_air):
        y = i * dy
        if i % 2 == 0:
            ax3.plot([0, L], [y, y], linewidth=2)  # left -> right
        else:
            ax3.plot([L, 0], [y, y], linewidth=2)  # right -> left

        # U-bend between pass i and i+1
        if i < n_passes_air - 1:
            y2 = (i + 1) * dy
            if i % 2 == 0:
                arc = Arc((L, (y + y2) / 2), width=0.9, height=0.9, angle=0, theta1=270, theta2=90, linewidth=2)
            else:
                arc = Arc((0, (y + y2) / 2), width=0.9, height=0.9, angle=0, theta1=90, theta2=270, linewidth=2)
            ax3.add_patch(arc)

    # mark inlet and outlet header connection points
    ax3.scatter([0], [0], s=60, zorder=5)
    ax3.text(0.05, 0, "IN (top/bottom header port)", va="center", fontsize=9)

    out_y = y_max
    out_x = 0 if same_side else L
    ax3.scatter([out_x], [out_y], s=60, zorder=5)
    ax3.text(out_x + (0.05 if out_x == 0 else -0.05), out_y,
             "OUT", va="center", ha=("left" if out_x == 0 else "right"), fontsize=9)

    ax3.set_title("View 3: One circuit serpentine\nHeaders same side if even passes; opposite if odd")
    ax3.text(0.0, -1.05, f"N_passes_air={n_passes_air}, U-bends={n_passes_air-1}", fontsize=9, ha="left")
    ax3.text(0.0, -1.35, f"L_straight={L_straight_m:.3f} m (header plane → U-bend start)", fontsize=9, ha="left")
    ax3.text(0.0, -1.65, f"Outlet header location: {hdr_out_side}", fontsize=9, ha="left")

    ax3.set_xlim(-0.3, L + 0.3)
    ax3.set_ylim(-2.0, y_max + 1.0)
    ax3.set_aspect("auto")
    ax3.axis("off")

    fig.tight_layout()
    return fig, tubes_across, tubes_high


def fig_to_png_bytes(fig) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return bio.getvalue()


# ============================================================
# Coil layout helpers
# ============================================================
def compute_circuit_distribution(total_tubes: int, circuits: int) -> pd.DataFrame:
    circuits = max(1, int(circuits))
    total_tubes = max(0, int(total_tubes))
    base = total_tubes // circuits
    rem = total_tubes % circuits
    rows = []
    for i in range(circuits):
        n = base + (1 if i < rem else 0)
        rows.append([i + 1, n])
    df = pd.DataFrame(rows, columns=["Circuit#", "TubePieces"])
    df["Share_%"] = (df["TubePieces"] / max(1, total_tubes) * 100.0).round(2)
    return df


# ============================================================
# PDF helpers
# ============================================================
def _dict_to_table_data(d: dict, styles, key_col="Parameter", val_col="Value"):
    """Return a 2-col table where values wrap safely (prevents LayoutError)."""
    normal = styles["BodyText"]
    header = styles["Heading6"]
    data = [[Paragraph(f"<b>{key_col}</b>", header), Paragraph(f"<b>{val_col}</b>", header)]]
    for k, v in d.items():
        ks = str(k)
        vs = str(v)
        data.append([Paragraph(ks, normal), Paragraph(vs, normal)])
    return data


def dataframe_to_pdf_table(df: pd.DataFrame, max_rows: int = 60):
    if df is None or len(df) == 0:
        return None
    df_show = df.copy()
    if len(df_show) > max_rows:
        df_show = df_show.head(max_rows)
    for c in df_show.columns:
        if pd.api.types.is_numeric_dtype(df_show[c]):
            df_show[c] = df_show[c].astype(float).round(4)
    return [list(df_show.columns)] + df_show.astype(str).values.tolist()


def build_pdf_report(
    title: str,
    inputs: dict,
    outputs: dict,
    intermediates: dict,
    layout_summary: dict,
    circuit_df: pd.DataFrame,
    df_profile: pd.DataFrame,
    schematic_png: bytes | None,
    include_profile: bool,
    profile_rows: int,
    include_schematic: bool
) -> bytes:
    """Build a PDF report that won't crash with ReportLab LayoutError.

    Fixes:
      - Wraps dict values with Paragraph (prevents unbreakable long strings)
      - Uses LongTable for big tables (allows page splitting)
      - Scales schematic image to fit page frame (caps max height)
    """
    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    story = []
    story.append(Paragraph(title, h1))
    story.append(Spacer(1, 6))

    def add_section(heading: str, d: dict):
        story.append(Paragraph(heading, h2))
        story.append(Spacer(1, 4))
        data = _dict_to_table_data(d, styles)
        tbl = LongTable(data, colWidths=[60 * mm, 105 * mm], repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 10))

    add_section("Inputs", inputs)

    if include_schematic and schematic_png:
        story.append(Paragraph("Coil schematic (definition sketch)", h2))
        story.append(Spacer(1, 4))

        img_reader = ImageReader(io.BytesIO(schematic_png))
        iw, ih = img_reader.getSize()

        max_w = 180 * mm
        max_h = 110 * mm  # critical: cap height to avoid LayoutError

        scale = min(max_w / iw, max_h / ih)
        img = RLImage(io.BytesIO(schematic_png), width=iw * scale, height=ih * scale)
        story.append(img)
        story.append(Spacer(1, 10))

    add_section("Key Outputs", outputs)
    add_section("Intermediate Parameters (debug)", intermediates)
    add_section("Coil Layout Summary", layout_summary)

    if circuit_df is not None and len(circuit_df) > 0:
        story.append(Paragraph("Circuit Distribution (approx.)", h2))
        story.append(Spacer(1, 4))

        df_show = circuit_df.copy()
        for c in df_show.columns:
            if pd.api.types.is_numeric_dtype(df_show[c]):
                df_show[c] = df_show[c].astype(float).round(4)

        data = [list(df_show.columns)] + df_show.astype(str).values.tolist()
        tbl = LongTable(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 10))

    if include_profile and df_profile is not None and len(df_profile) > 0:
        story.append(PageBreak())
        story.append(Paragraph("Merkel Marching Profile", h2))
        story.append(Paragraph(
            f"Showing last {min(profile_rows, len(df_profile))} rows (of {len(df_profile)} total).",
            normal
        ))
        story.append(Spacer(1, 6))

        df_show = df_profile.tail(profile_rows).copy()
        for c in df_show.columns:
            if pd.api.types.is_numeric_dtype(df_show[c]):
                df_show[c] = df_show[c].astype(float).round(4)

        data = [list(df_show.columns)] + df_show.astype(str).values.tolist()
        tbl = LongTable(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(tbl)

    doc.build(story)
    return buf.getvalue()


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Evaporative Cooler Coil Designer", layout="wide")
st.title("Forced-Draft Evaporative Fluid Cooler — Coil + Fan + Hydraulics (Glycol + CoolProp + PDF)")

if not COOLPROP_OK:
    st.error("CoolProp is not installed. Use CoolProp==6.7.0 in requirements.txt for Streamlit Python 3.13.")
    st.stop()

st.caption(
    "Flexible coil geometry inputs (rows, circuits, pitches, face size, tube length). "
    "Adds 3 schematics: plan + side + one-circuit serpentine with even/odd header rule. "
    "Sizes required wetted area using a Merkel-style enthalpy marching model (calibratable K), "
    "checks provided coil area from geometry, estimates tube ΔP and fan power, and generates a PDF report."
)

tab1, tab2, tab3 = st.tabs(["Inputs + Schematic", "Results", "PDF Report"])

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
        user_flow = st.number_input(
            "User flow (m³/h)",
            value=15.0,
            step=0.5,
            disabled=(flow_mode != "User flow (m³/h)")
        )

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

        rows_depth = st.number_input("Number of rows (DEPTH, airflow direction)", min_value=1, value=6, step=1)
        vert_pitch_mm = st.number_input("Vertical pitch (mm)", min_value=15.0, value=50.0, step=1.0)
        horiz_pitch_mm = st.number_input("Horizontal pitch (mm)", min_value=15.0, value=50.0, step=1.0)

        face_W_m = st.number_input("Coil face width (m)", min_value=0.3, value=1.5, step=0.1)
        face_H_m = st.number_input("Coil face height (m)", min_value=0.3, value=1.5, step=0.1)

        tube_length_m = st.number_input("Tube straight length per piece (m)", min_value=0.3, value=1.5, step=0.1)
        circuits = st.number_input("Number of parallel circuits", min_value=1, value=8, step=1)

        # Serpentine schematic inputs (one-circuit view)
        n_passes_air = st.number_input("Serpentine passes per circuit (rows in air path)", min_value=2, value=12, step=1)
        L_straight_m = st.number_input("Straight pass length for serpentine view (m)", min_value=0.2, value=1.5, step=0.1)

        hdr_in_mm = st.number_input("Inlet header diameter (mm)", min_value=25.0, value=80.0, step=5.0)
        hdr_out_mm = st.number_input("Outlet header diameter (mm)", min_value=25.0, value=80.0, step=5.0)

        K_minor = st.number_input("Minor-loss coefficient per circuit (bends+entry/exit)", min_value=0.0, value=3.0, step=0.5)

    st.subheader("Coil schematic (3 views)")
    fig, tubes_across, tubes_high = draw_three_views(
        face_W_m=face_W_m,
        face_H_m=face_H_m,
        rows_depth=int(rows_depth),
        horiz_pitch_mm=float(horiz_pitch_mm),
        vert_pitch_mm=float(vert_pitch_mm),
        Do_mm=float(Do_mm),
        n_passes_air=int(n_passes_air),
        L_straight_m=float(L_straight_m)
    )
    st.pyplot(fig, use_container_width=True)
    schematic_png = fig_to_png_bytes(fig)

    st.caption(
        f"Pitch-based counts: tubes_across≈{tubes_across}, tubes_high≈{tubes_high}. "
        f"Total straight tube pieces (for area) = rows_depth × tubes_across × tubes_high."
    )

run = st.button("Run Design & Check", type="primary")

if "results" not in st.session_state:
    st.session_state["results"] = None

with tab2:
    st.subheader("Results")

    if not run:
        st.info("Enter inputs in the Inputs tab and click **Run Design & Check**.")
    else:
        T_mean = 0.5 * (Tw_in + Tw_out)
        proc_fluid = make_process_fluid(fluid_choice, glycol_pct)
        cp, rho, mu, k_fluid = fluid_props(proc_fluid, T_mean, P_kpa)

        w_in = humidity_ratio_from_tdb_twb(Tdb, Twb, P_kpa)
        rho_air = air_density_kg_per_m3(Tdb, w_in, P_kpa)

        if air_mode == "Estimate from Δh (kJ/kg_da)":
            m_air_est = Q_kw / dh_assumed
            Vdot_air = (m_air_est / rho_air) * 3600.0

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

        if flow_mode == "User flow (m³/h)":
            flow_m3_h = user_flow
            m_w = (flow_m3_h * rho) / 3600.0
            Q_check = m_w * cp * (Tw_in - Tw_out)
        else:
            m_w = m_w_auto
            flow_m3_h = (m_w * 3600.0) / max(rho, 1e-9)
            Q_check = Q_kw

        # Tube counts from pitch & face
        tubes_per_row = tubes_across
        tubes_in_height = tubes_high
        tubes_per_layer = tubes_per_row * tubes_in_height
        total_tubes = tubes_per_layer * int(rows_depth)

        # Total tube straight length
        L_total = total_tubes * tube_length_m

        # External area provided (bare tube)
        Do_m = Do_mm / 1000.0
        A_provided = math.pi * Do_m * L_total

        # Tube internal diameter
        Di_mm = max(0.5, Do_mm - 2.0 * t_mm)
        Di_m = Di_mm / 1000.0

        # Circuit length approximation (straight-only estimate)
        L_circuit = L_total / max(int(circuits), 1)

        # Internal velocity per circuit
        flow_m3_s = flow_m3_h / 3600.0
        flow_per_circuit_m3_s = flow_m3_s / max(int(circuits), 1)
        flow_per_circuit_m3_h = flow_per_circuit_m3_s * 3600.0

        A_id = math.pi * (Di_m ** 2) / 4.0
        v_int = flow_per_circuit_m3_s / max(A_id, 1e-12)

        Re = reynolds(rho, v_int, Di_m, mu)
        Pr = prandtl(cp, mu, k_fluid)
        Nu = nusselt_dittus_boelter(Re, Pr, heating=False)
        h_i = Nu * k_fluid / max(Di_m, 1e-12)

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

        margin_pct = (A_provided / max(A_req, 1e-9) - 1.0) * 100.0

        layout_summary = {
            "Rows (depth)": int(rows_depth),
            "Face width (m)": f"{face_W_m:.3f}",
            "Face height (m)": f"{face_H_m:.3f}",
            "Horizontal pitch (mm)": f"{horiz_pitch_mm:.1f}",
            "Vertical pitch (mm)": f"{vert_pitch_mm:.1f}",
            "Tubes across width": int(tubes_per_row),
            "Tubes in height": int(tubes_in_height),
            "Tubes per layer (W×H)": int(tubes_per_layer),
            "Total tube pieces (layers×rows)": int(total_tubes),
            "Tube length per piece (m)": f"{tube_length_m:.3f}",
            "Total straight tube length (m)": f"{L_total:.2f}",
            "Circuits": int(circuits),
            "Flow total (m³/h)": f"{flow_m3_h:.3f}",
            "Flow per circuit (m³/h)": f"{flow_per_circuit_m3_h:.3f}",
            "Header inlet diameter (mm)": f"{hdr_in_mm:.1f}",
            "Header outlet diameter (mm)": f"{hdr_out_mm:.1f}",
            "Serpentine passes per circuit (view)": int(n_passes_air),
            "Serpentine straight length (view) (m)": f"{L_straight_m:.3f}",
            "Header same-side rule": "Even passes → same side; Odd passes → opposite sides",
        }

        circuit_df = compute_circuit_distribution(total_tubes=total_tubes, circuits=circuits)
        circuit_df["TubeLength_m_est"] = (circuit_df["TubePieces"] * tube_length_m).round(3)
        circuit_df["CircuitFlow_m3_h_est"] = float(flow_per_circuit_m3_h)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duty (kW)", f"{Q_kw:.2f}")
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
        c12.metric("Total tube pieces (straight)", f"{total_tubes:d}")
        st.caption("Tube counts use floor(face dimension / pitch). Adjust face sizes/pitches to hit desired count.")

        st.write("### Tube-side Hydraulics (per circuit, first estimate)")
        c13, c14, c15, c16 = st.columns(4)
        c13.metric("Tube ID (mm)", f"{Di_mm:.2f}")
        c14.metric("Internal velocity (m/s)", f"{v_int:.2f}")
        c15.metric("Re", f"{Re:.0f}")
        c16.metric("ΔP per circuit (kPa)", f"{dp_pa/1000.0:.2f}")

        st.write("### Tube-side Heat Transfer (internal)")
        c17, c18, c19 = st.columns(3)
        c17.metric("Pr", f"{Pr:.2f}")
        c18.metric("Nu", f"{Nu:.1f}")
        c19.metric("hᵢ (W/m²·K)", f"{h_i:.0f}")

        st.write("### Header velocity sanity check")
        c20, c21 = st.columns(2)
        c20.metric("Inlet header velocity (m/s)", f"{v_hdr_in:.2f}")
        c21.metric("Outlet header velocity (m/s)", f"{v_hdr_out:.2f}")
        st.caption("Typical practice: headers ~1–2.5 m/s (context-dependent).")

        st.write("### Air & Fan")
        c22, c23, c24, c25 = st.columns(4)
        c22.metric("Airflow (m³/h)", f"{Vdot_air:.0f}")
        c23.metric("Air density (kg/m³)", f"{rho_air_calc:.2f}")
        c24.metric("Fan static (Pa)", f"{dP_fan:.0f}")
        c25.metric("Fan shaft power (kW)", f"{fan_kw:.2f}")

        with st.expander("Coil layout summary (debug)"):
            st.json(layout_summary)

        with st.expander("Circuit distribution table (debug)"):
            st.dataframe(circuit_df, use_container_width=True)

        with st.expander("Merkel marching profile (last 50 rows)"):
            st.dataframe(df_profile.tail(50), use_container_width=True)

        st.session_state["results"] = {
            "inputs_raw": {
                "unitQ": unitQ,
                "Q_in": Q_in,
                "Q_kw": Q_kw,
                "Tw_in": Tw_in,
                "Tw_out": Tw_out,
                "fluid_choice": fluid_choice,
                "glycol_pct": glycol_pct,
                "flow_mode": flow_mode,
                "user_flow": user_flow,
                "P_kpa": P_kpa,
                "Tdb": Tdb,
                "Twb": Twb,
                "air_mode": air_mode,
                "Vdot_air": Vdot_air,
                "dh_assumed": dh_assumed,
                "dP_fan": dP_fan,
                "eta_fan": eta_fan,
                "K": K,
                "dA_step": dA_step,
                "max_area": max_area,
                "tube_mat": tube_mat,
                "Do_mm": Do_mm,
                "t_mm": t_mm,
                "rows_depth": rows_depth,
                "vert_pitch_mm": vert_pitch_mm,
                "horiz_pitch_mm": horiz_pitch_mm,
                "face_W_m": face_W_m,
                "face_H_m": face_H_m,
                "tube_length_m": tube_length_m,
                "circuits": circuits,
                "hdr_in_mm": hdr_in_mm,
                "hdr_out_mm": hdr_out_mm,
                "K_minor": K_minor,
                "n_passes_air": int(n_passes_air),
                "L_straight_m": float(L_straight_m),
            },
            "proc_fluid": proc_fluid,
            "cp_kJ_kgK": cp,
            "rho_kg_m3": rho,
            "mu_Pa_s": mu,
            "k_W_mK": k_fluid,
            "A_req_m2": A_req,
            "A_provided_m2": A_provided,
            "area_margin_pct": margin_pct,
            "flow_m3_h": flow_m3_h,
            "m_w_kg_s": m_w,
            "m_air_kg_s": m_air,
            "w_in": w_in_calc,
            "rho_air": rho_air_calc,
            "Di_mm": Di_mm,
            "v_int_m_s": v_int,
            "Re": Re,
            "Pr": Pr,
            "Nu": Nu,
            "h_i_W_m2K": h_i,
            "dp_circuit_kPa": dp_pa/1000.0,
            "v_hdr_in_m_s": v_hdr_in,
            "v_hdr_out_m_s": v_hdr_out,
            "fan_power_kw": fan_kw,
            "Q_check_kW": Q_check,
            "layout_summary": layout_summary,
            "circuit_df": circuit_df,
            "df_profile": df_profile,
            "schematic_png": schematic_png,
        }

with tab3:
    st.subheader("PDF Report Export")

    res = st.session_state.get("results", None)
    if not res:
        st.info("Run the calculation first (Results tab).")
    else:
        include_profile = st.checkbox("Include Merkel marching table in PDF", value=True)
        include_schematic = st.checkbox("Include schematic image in PDF", value=True)
        profile_rows = st.slider("How many marching rows to include", min_value=20, max_value=500, value=120, step=20)

        ir = res["inputs_raw"]

        inputs = {
            "Heat rejection (kW)": f"{ir['Q_kw']:.3f}",
            "Process fluid": res["proc_fluid"],
            "Tw in (°C)": ir["Tw_in"],
            "Tw out (°C)": ir["Tw_out"],
            "Air DB (°C)": ir["Tdb"],
            "Air WB (°C)": ir["Twb"],
            "Pressure (kPa)": ir["P_kpa"],
            "Airflow (m³/h)": f"{ir['Vdot_air']:.0f}",
            "K (kg/s·m²)": ir["K"],
            "dA step (m²)": ir["dA_step"],
            "Max area (m²)": ir["max_area"],
            "Fan ΔP (Pa)": ir["dP_fan"],
            "Fan η": ir["eta_fan"],
            "Tube material": ir["tube_mat"],
            "Tube OD (mm)": ir["Do_mm"],
            "Tube thickness (mm)": ir["t_mm"],
            "Rows depth": ir["rows_depth"],
            "Vertical pitch (mm)": ir["vert_pitch_mm"],
            "Horizontal pitch (mm)": ir["horiz_pitch_mm"],
            "Face W (m)": ir["face_W_m"],
            "Face H (m)": ir["face_H_m"],
            "Tube length per piece (m)": ir["tube_length_m"],
            "Circuits": ir["circuits"],
            "Header in (mm)": ir["hdr_in_mm"],
            "Header out (mm)": ir["hdr_out_mm"],
            "Minor loss K per circuit": ir["K_minor"],
            "Serpentine passes (view)": ir["n_passes_air"],
            "Serpentine straight length (view) (m)": ir["L_straight_m"],
        }

        outputs = {
            "Required coil area (m²)": f"{res['A_req_m2']:.3f}",
            "Provided coil area (m²)": f"{res['A_provided_m2']:.3f}",
            "Area margin (%)": f"{res['area_margin_pct']:.2f}",
            "Process flow (m³/h)": f"{res['flow_m3_h']:.3f}",
            "Fan power (kW)": f"{res['fan_power_kw']:.3f}",
            "Tube ΔP per circuit (kPa)": f"{res['dp_circuit_kPa']:.3f}",
            "Energy check Q_check (kW)": f"{res['Q_check_kW']:.3f}",
        }

        intermediates = {
            "cp (kJ/kg·K)": f"{res['cp_kJ_kgK']:.6f}",
            "ρ (kg/m³)": f"{res['rho_kg_m3']:.3f}",
            "μ (mPa·s)": f"{res['mu_Pa_s']*1000.0:.6f}",
            "k (W/m·K)": f"{res['k_W_mK']:.6f}",
            "m_w (kg/s)": f"{res['m_w_kg_s']:.6f}",
            "m_air (kg/s)": f"{res['m_air_kg_s']:.6f}",
            "w_in (kg/kg_da)": f"{res['w_in']:.6f}",
            "ρ_air (kg/m³)": f"{res['rho_air']:.6f}",
            "Tube ID (mm)": f"{res['Di_mm']:.3f}",
            "v_int (m/s)": f"{res['v_int_m_s']:.6f}",
            "Re": f"{res['Re']:.0f}",
            "Pr": f"{res['Pr']:.4f}",
            "Nu": f"{res['Nu']:.2f}",
            "h_i (W/m²·K)": f"{res['h_i_W_m2K']:.1f}",
            "Header v_in (m/s)": f"{res['v_hdr_in_m_s']:.3f}",
            "Header v_out (m/s)": f"{res['v_hdr_out_m_s']:.3f}",
        }

        pdf_bytes = build_pdf_report(
            title="Evaporative Fluid Cooler Coil Design Report",
            inputs=inputs,
            outputs=outputs,
            intermediates=intermediates,
            layout_summary=res["layout_summary"],
            circuit_df=res["circuit_df"],
            df_profile=res["df_profile"],
            schematic_png=res.get("schematic_png", None),
            include_profile=include_profile,
            profile_rows=profile_rows,
            include_schematic=include_schematic
        )

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="evap_fluid_cooler_report.pdf",
            mime="application/pdf"
        )

        st.caption("Upload the PDF here anytime; it contains every intermediate value used by the calculation.")
