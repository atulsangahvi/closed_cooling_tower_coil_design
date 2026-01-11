import math
import io
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# Plotting (GA + coil detail)
# -------------------------
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle as MplCircle

# -------------------------
# PDF generation (ReportLab)
# -------------------------
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Image as RLImage, LongTable, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# -------------------------
# CoolProp import
# -------------------------
try:
    from CoolProp.CoolProp import PropsSI
    COOLPROP_OK = True
except Exception:
    COOLPROP_OK = False


# ============================================================
# Psychrometrics (engineering approximations)
# ============================================================
def p_ws_kpa(T_c: float) -> float:
    """Saturation vapor pressure over water (kPa), good for ~0–60°C."""
    return 0.61094 * math.exp((17.625 * T_c) / (T_c + 243.04))


def humidity_ratio_from_tdb_twb(T_db: float, T_wb: float, P_kpa: float = 101.325) -> float:
    """Humidity ratio w (kg/kg_da) from dry bulb and wet bulb (psychrometric approximation)."""
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
    return p_da / (R_da * T_k) + p_w / (R_wv * T_k)


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
        raise RuntimeError("CoolProp not available.")
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
    max_area_m2: float = 3000.0,
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

        dQ = K_kg_s_m2 * dA_step_m2 * drive  # kW (kJ/s)
        h_a_new = h_a + dQ / max(1e-12, m_air)
        Tw_new = Tw - dQ / max(1e-12, (m_w * cp_kj_kgK))

        A += dA_step_m2
        rows.append([step, A, Tw, hs, h_a, drive, dQ, m_air, m_w])
        step += 1
        Tw, h_a = Tw_new, h_a_new

        if step > 400000:
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
# Utility
# ============================================================
def fig_to_png_bytes(fig) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return bio.getvalue()


def compute_circuit_distribution(total_tube_pieces: int, circuits: int) -> pd.DataFrame:
    circuits = max(1, int(circuits))
    total_tube_pieces = max(0, int(total_tube_pieces))
    base = total_tube_pieces // circuits
    rem = total_tube_pieces % circuits
    rows = []
    for i in range(circuits):
        n = base + (1 if i < rem else 0)
        rows.append([i + 1, n])
    df = pd.DataFrame(rows, columns=["Circuit#", "TubePieces"])
    df["Share_%"] = (df["TubePieces"] / max(1, total_tube_pieces) * 100.0).round(2)
    return df


# ============================================================
# GA + Coil detail drawing
# ============================================================
def _dim_arrow(ax, x1, y1, x2, y2, text, text_offset=(0, 0), fontsize=9):
    """Dimension arrow helper.

    Important for Streamlit: Streamlit renders figures using `bbox_inches="tight"`.
    If we place annotations far outside the axes limits (negative coordinates based
    on large user-entered dimensions), the tight bounding box can explode and
    Matplotlib will raise "Image size too large".

    So we clip annotations to the axes.
    """
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x2, y2),
        arrowprops=dict(arrowstyle="<->", linewidth=1.2),
        clip_on=True,
    )
    tx = (x1 + x2) / 2 + text_offset[0]
    ty = (y1 + y2) / 2 + text_offset[1]
    ax.text(tx, ty, text, ha="center", va="center", fontsize=fontsize, clip_on=True)


def draw_ga_and_coil_detail(
    casing_L: float, casing_W: float, casing_H: float,
    face_W: float, face_H: float,
    rows_depth: int,
    Pt_mm: float, Pv_mm: float, Pr_mm: float,
    Do_mm: float,
    fan_diam: float,
    eliminator_thk: float,
    sump_depth: float,
    nozzle_count: int,
):
    """
    Produces a clean GA-style figure (Top/Side/End) plus two coil-detail insets:
      - Detail A: WIDTH x DEPTH (Pt, Pr, Do)
      - Detail B: HEIGHT x DEPTH (Pv, Pr, Do)
    Draws a representative subset of tubes (max 10x10) so it's readable.
    """
    Pt = Pt_mm / 1000.0
    Pv = Pv_mm / 1000.0
    Pr = Pr_mm / 1000.0
    Do = Do_mm / 1000.0

    tubes_across = max(1, int(math.floor(face_W / max(Pt, 1e-9))))
    tubes_high = max(1, int(math.floor(face_H / max(Pv, 1e-9))))
    rows_depth = max(1, int(rows_depth))

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_end = fig.add_subplot(gs[0, 2])
    ax_detA = fig.add_subplot(gs[1, 0:2])
    ax_detB = fig.add_subplot(gs[1, 2])

    # ---------------- Top view (plan) ----------------
    ax_top.set_title("GA Top View (Plan)")
    ax_top.add_patch(Rectangle((0, 0), casing_L, casing_W, fill=False, linewidth=2))
    # Coil footprint (assume placed near one end for forced draft)
    coil_x = casing_L * 0.20
    coil_y = (casing_W - face_W) / 2
    coil_L = max(0.2, casing_L * 0.45)
    ax_top.add_patch(Rectangle((coil_x, coil_y), coil_L, face_W, fill=False, linewidth=2, linestyle="--"))
    ax_top.text(coil_x + coil_L/2, coil_y + face_W/2, "COIL FOOTPRINT", ha="center", va="center", fontsize=9)

    # Spray header line above coil footprint
    ax_top.plot([coil_x, coil_x + coil_L], [coil_y + face_W*0.80, coil_y + face_W*0.80], linewidth=2)
    ax_top.text(coil_x, coil_y + face_W*0.86, f"Spray header (≈{nozzle_count} nozzles)", fontsize=8, ha="left")

    # Fan section location
    fan_x = casing_L * 0.80
    ax_top.add_patch(Rectangle((fan_x, 0), casing_L - fan_x, casing_W, fill=False, linewidth=1.5))
    ax_top.text((fan_x + casing_L)/2, casing_W/2, "FAN SECTION", ha="center", va="center", fontsize=9)

    _dim_arrow(ax_top, casing_L*0.05, casing_W*0.06, casing_L*0.95, casing_W*0.06, f"L={casing_L:.2f} m", text_offset=(0, casing_W*0.03))
    _dim_arrow(ax_top, casing_L*0.06, casing_W*0.05, casing_L*0.06, casing_W*0.95, f"W={casing_W:.2f} m", text_offset=(casing_L*0.03, 0))
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.axis("off")

    # ---------------- Side view (elevation) ----------------
    ax_side.set_title("GA Side View (Elevation)")
    ax_side.add_patch(Rectangle((0, 0), casing_L, casing_H, fill=False, linewidth=2))
    # Sump line
    ax_side.plot([0, casing_L], [sump_depth, sump_depth], linewidth=1.5)
    ax_side.text(casing_L*0.02, sump_depth + casing_H*0.02, "SUMP", fontsize=8)

    # Coil block in elevation
    coil_z = sump_depth + casing_H*0.10
    coil_h = min(face_H, casing_H*0.55)
    coil_L_side = max(0.2, casing_L*0.45)
    ax_side.add_patch(Rectangle((coil_x, coil_z), coil_L_side, coil_h, fill=False, linewidth=2, linestyle="--"))
    ax_side.text(coil_x + coil_L_side/2, coil_z + coil_h/2, "COIL", ha="center", va="center", fontsize=10)

    # Drift eliminator block above coil
    elim_h = casing_H*0.15
    ax_side.add_patch(Rectangle((coil_x, coil_z + coil_h), coil_L_side, elim_h, fill=False, linewidth=1.6))
    ax_side.text(coil_x + coil_L_side/2, coil_z + coil_h + elim_h/2, "DRIFT\nELIMINATOR", ha="center", va="center", fontsize=8)

    # Fan circle in elevation (forced draft)
    fan_center = (fan_x + (casing_L - fan_x)/2, casing_H*0.55)
    ax_side.add_patch(MplCircle(fan_center, radius=fan_diam/2, fill=False, linewidth=2))
    ax_side.text(fan_center[0], fan_center[1], f"FAN\nØ{fan_diam:.2f} m", ha="center", va="center", fontsize=8)

    _dim_arrow(ax_side, casing_L*0.05, casing_H*0.06, casing_L*0.95, casing_H*0.06, f"L={casing_L:.2f} m", text_offset=(0, casing_H*0.03))
    _dim_arrow(ax_side, casing_L*0.06, casing_H*0.05, casing_L*0.06, casing_H*0.95, f"H={casing_H:.2f} m", text_offset=(casing_L*0.03, 0))
    ax_side.set_aspect("equal", adjustable="box")
    ax_side.axis("off")

    # ---------------- End view ----------------
    ax_end.set_title("GA End View")
    ax_end.add_patch(Rectangle((0, 0), casing_W, casing_H, fill=False, linewidth=2))
    # coil face opening
    face_x = (casing_W - face_W) / 2
    face_y = sump_depth + casing_H*0.10
    ax_end.add_patch(Rectangle((face_x, face_y), face_W, min(face_H, casing_H*0.55),
                               fill=False, linewidth=2, linestyle="--"))
    ax_end.text(casing_W/2, face_y + min(face_H, casing_H*0.55)/2, "COIL FACE", ha="center", va="center", fontsize=9)

    _dim_arrow(ax_end, casing_W*0.05, casing_H*0.06, casing_W*0.95, casing_H*0.06, f"W={casing_W:.2f} m", text_offset=(0, casing_H*0.03))
    _dim_arrow(ax_end, casing_W*0.06, casing_H*0.05, casing_W*0.06, casing_H*0.95, f"H={casing_H:.2f} m", text_offset=(casing_W*0.03, 0))
    ax_end.set_aspect("equal", adjustable="box")
    ax_end.axis("off")

    # ---------------- Detail A: WIDTH x DEPTH ----------------
    ax_detA.set_title("Coil Detail A (Plan Slice): WIDTH × DEPTH (Pitch Pt, Pr)")
    max_cols = min(10, tubes_across)
    max_rows = min(10, rows_depth)
    # represent a slice; spacing in mm for clarity
    x0, y0 = 0.0, 0.0
    for r in range(max_rows):
        for c in range(max_cols):
            cx = x0 + c * Pt_mm
            cy = y0 + r * Pr_mm
            ax_detA.add_patch(MplCircle((cx, cy), radius=Do_mm/2, fill=False, linewidth=1.0))
    # dimension callouts (Pt and Pr)
    if max_cols >= 2:
        _dim_arrow(ax_detA, x0, -Do_mm*1.5, x0 + Pt_mm, -Do_mm*1.5, f"Pt={Pt_mm:.1f} mm", text_offset=(0, -Do_mm*0.9), fontsize=9)
    if max_rows >= 2:
        _dim_arrow(ax_detA, -Do_mm*1.5, y0, -Do_mm*1.5, y0 + Pr_mm, f"Pr={Pr_mm:.1f} mm", text_offset=(-Do_mm*1.4, 0), fontsize=9)

    ax_detA.text(
        x0, y0 + max_rows*Pr_mm + Do_mm*1.5,
        f"Computed: tubes_across≈{tubes_across}, depth_rows={rows_depth}, tube OD Do={Do_mm:.1f} mm",
        fontsize=10, ha="left"
    )
    ax_detA.set_aspect("equal", adjustable="box")
    ax_detA.axis("off")

    # ---------------- Detail B: HEIGHT x DEPTH ----------------
    ax_detB.set_title("Coil Detail B (Side Slice): HEIGHT × DEPTH (Pitch Pv, Pr)")
    max_h = min(10, tubes_high)
    max_rows2 = min(10, rows_depth)
    x0b, y0b = 0.0, 0.0
    for r in range(max_rows2):
        for k in range(max_h):
            cx = x0b + r * Pr_mm
            cy = y0b + k * Pv_mm
            ax_detB.add_patch(MplCircle((cx, cy), radius=Do_mm/2, fill=False, linewidth=1.0))
    if max_h >= 2:
        _dim_arrow(ax_detB, -Do_mm*1.5, y0b, -Do_mm*1.5, y0b + Pv_mm, f"Pv={Pv_mm:.1f} mm",
                   text_offset=(-Do_mm*1.4, 0), fontsize=9)
    if max_rows2 >= 2:
        _dim_arrow(ax_detB, x0b, -Do_mm*1.5, x0b + Pr_mm, -Do_mm*1.5, f"Pr={Pr_mm:.1f} mm",
                   text_offset=(0, -Do_mm*0.9), fontsize=9)

    ax_detB.text(
        x0b, y0b + max_h*Pv_mm + Do_mm*1.5,
        f"Computed: tubes_high≈{tubes_high}",
        fontsize=10, ha="left"
    )
    ax_detB.set_aspect("equal", adjustable="box")
    ax_detB.axis("off")

    return fig, tubes_across, tubes_high


# ============================================================
# PDF report (safe: wraps text, splits tables, scales images)
# ============================================================
def _dict_to_table_data(d: dict, styles, key_col="Parameter", val_col="Value"):
    normal = styles["BodyText"]
    header = styles["Heading6"]
    data = [[Paragraph(f"<b>{key_col}</b>", header), Paragraph(f"<b>{val_col}</b>", header)]]
    for k, v in d.items():
        data.append([Paragraph(str(k), normal), Paragraph(str(v), normal)])
    return data


def build_pdf_report(
    title: str,
    inputs: dict,
    outputs: dict,
    intermediates: dict,
    layout_summary: dict,
    circuit_df: pd.DataFrame,
    df_profile: pd.DataFrame,
    drawing_png: bytes | None,
    include_profile: bool,
    profile_rows: int,
    include_drawing: bool,
):
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

    def add_section(heading, d):
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

    if include_drawing and drawing_png:
        story.append(Paragraph("GA + Coil Detail Drawing", h2))
        story.append(Spacer(1, 4))
        img_reader = ImageReader(io.BytesIO(drawing_png))
        iw, ih = img_reader.getSize()
        max_w = 180 * mm
        max_h = 120 * mm
        scale = min(max_w / iw, max_h / ih)
        story.append(RLImage(io.BytesIO(drawing_png), width=iw * scale, height=ih * scale))
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
st.set_page_config(page_title="Evaporative Fluid Cooler Coil Designer", layout="wide")
st.title("Forced-Draft Evaporative Fluid Cooler — Coil + Fan + Hydraulics (CoolProp + PDF + GA Drawing)")

if not COOLPROP_OK:
    st.error("CoolProp is not installed. For Streamlit Python 3.13, use CoolProp==6.7.0 in requirements.txt.")
    st.stop()

tabs = st.tabs(["Inputs + Drawing", "Results", "PDF Report"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)

    with c1:
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

    with c2:
        st.subheader("Ambient Air (Design)")
        P_kpa = st.number_input("Barometric pressure, kPa", value=101.325, step=0.1)
        Tdb = st.number_input("Entering air Dry Bulb, °C", value=42.0, step=0.5)
        Twb = st.number_input("Entering air Wet Bulb, °C", value=30.0, step=0.5)

        air_mode = st.radio("Airflow mode", ["User airflow (m³/h)", "Estimate from Δh (kJ/kg_da)"], horizontal=True)
        if air_mode == "User airflow (m³/h)":
            Vdot_air = st.number_input("Airflow through unit, m³/h", min_value=1000.0, value=22000.0, step=500.0)
            dh_assumed = 15.0
        else:
            dh_assumed = st.number_input("Assumed air enthalpy rise Δh, kJ/kg_da", min_value=5.0, value=15.0, step=1.0)
            Vdot_air = None

        st.subheader("Fan")
        dP_fan = st.number_input("Fan total static pressure, Pa", min_value=50.0, value=200.0, step=10.0)
        eta_fan = st.number_input("Fan+motor+drive efficiency (0–1)", min_value=0.2, max_value=0.85, value=0.60, step=0.05)

    with c3:
        st.subheader("Merkel Transfer (Calibratable)")
        K = st.number_input("K coefficient (kg/s·m²)", min_value=0.0001, max_value=0.01, value=0.0015, step=0.0001, format="%.4f")
        dA_step = st.number_input("Marching dA step (m²)", min_value=0.01, value=0.10, step=0.01)
        max_area = st.number_input("Max area limit (m²) safety", min_value=50.0, value=3000.0, step=50.0)

        st.subheader("Tube Bundle Geometry (used for area + hydraulics)")
        Do_mm = st.number_input("Tube OD Do (mm)", min_value=6.0, value=25.4, step=0.5)
        t_mm = st.number_input("Tube thickness (mm)", min_value=0.5, value=2.5, step=0.1)

        rows_depth = st.number_input("Rows in DEPTH (air crosses this many rows)", min_value=1, value=6, step=1)
        Pt_mm = st.number_input("Tube-to-tube pitch Pt across WIDTH (mm)", min_value=15.0, value=50.0, step=1.0)
        Pv_mm = st.number_input("Vertical pitch Pv (mm)", min_value=15.0, value=50.0, step=1.0)
        Pr_mm = st.number_input("Row-to-row pitch Pr in DEPTH (mm)", min_value=15.0, value=50.0, step=1.0)

        face_W_m = st.number_input("Coil face width (m)", min_value=0.3, value=1.5, step=0.1)
        face_H_m = st.number_input("Coil face height (m)", min_value=0.3, value=1.5, step=0.1)

        tube_length_m = st.number_input("Tube length per piece (m) [for area & material]", min_value=0.3, value=1.5, step=0.1)
        circuits = st.number_input("Number of parallel circuits", min_value=1, value=10, step=1)

        hdr_in_mm = st.number_input("Inlet header diameter (mm)", min_value=25.0, value=80.0, step=5.0)
        hdr_out_mm = st.number_input("Outlet header diameter (mm)", min_value=25.0, value=80.0, step=5.0)
        K_minor = st.number_input("Minor-loss coefficient per circuit (bends+entry/exit)", min_value=0.0, value=3.0, step=0.5)

        st.subheader("GA inputs (for drawing/quoting)")
        casing_L_m = st.number_input("Overall unit length L (m)", min_value=0.5, value=max(2.5, face_W_m * 2.0), step=0.1)
        casing_W_m = st.number_input("Overall unit width W (m)", min_value=0.3, value=max(1.2, face_W_m * 0.9), step=0.1)
        casing_H_m = st.number_input("Overall unit height H (m)", min_value=0.5, value=max(1.8, face_H_m * 1.2), step=0.1)
        fan_diam_m = st.number_input("Fan diameter (m)", min_value=0.2, value=0.9, step=0.05)
        eliminator_thk_m = st.number_input("Drift eliminator thickness (m)", min_value=0.02, value=0.15, step=0.01)
        sump_depth_m = st.number_input("Sump depth (m)", min_value=0.05, value=0.30, step=0.05)
        nozzle_count = st.number_input("Nozzle count (for drawing label)", min_value=0, value=12, step=1)

    st.divider()

    with st.expander("Notes (K meaning, typical ranges, tube lengths)", expanded=False):
        st.markdown(
            """
### Merkel coefficient **K** (units: **kg/s·m²**)
- This app uses a Merkel / enthalpy driving-force form:
  **dQ = K · dA · (hₛ(Tw) − hₐ)**  
  where **(hₛ − hₐ)** is in **kJ/kg dry air**, so **K must be kg/s·m²** to give **kJ/s = kW**.
- **K is not a W/m²·K heat-transfer coefficient.** It’s an overall **mass-transfer coefficient** on an air-mass basis.
- Practically, K lumps together spray wetting quality, droplet/film contact, air-side transfer, and overall evaporative effectiveness.
- In industry, K is usually **calibrated** from a known unit or vendor performance at similar spray + airflow conditions.

**Typical K ranges (rough design guidance):**
- Poor spray distribution / low wetting: **0.0005 – 0.0010**
- Spray over tube bundle (typical closed-circuit fluid cooler): **0.0010 – 0.0025**
- Optimized spray + very good wetting / assist media: **0.0020 – 0.0040**
- High-performance open-tower fill (not a closed-circuit coil): **0.0040 – 0.0070**

### Tube length definitions
**Tube length per piece (area + material + costing):**
- Physical straight tube piece length used to compute total tube length and external area:
  **A_provided = π·Do·(total_tube_pieces × tube_length_per_piece)**

**(If you later add serpentine circuit hydraulic detail) “straight pass length”:**
- Distance from header plane → start of U-bend for one pass in one circuit.
- Used for per-circuit hydraulic length; not the same thing as total tube metal length.
"""
        )

    st.subheader("GA + Coil Bundle Detail Drawing")
    fig, tubes_across, tubes_high = draw_ga_and_coil_detail(
        casing_L=casing_L_m, casing_W=casing_W_m, casing_H=casing_H_m,
        face_W=face_W_m, face_H=face_H_m,
        rows_depth=int(rows_depth),
        Pt_mm=float(Pt_mm), Pv_mm=float(Pv_mm), Pr_mm=float(Pr_mm),
        Do_mm=float(Do_mm),
        fan_diam=float(fan_diam_m),
        eliminator_thk=float(eliminator_thk_m),
        sump_depth=float(sump_depth_m),
        nozzle_count=int(nozzle_count),
    )
    st.pyplot(fig, use_container_width=True)
    drawing_png = fig_to_png_bytes(fig)

    st.markdown(
        f"""
**Computed tube counts used by the model (from face size and pitches):**
- tubes_across ≈ floor(W / Pt) = **{tubes_across}**
- tubes_high ≈ floor(H / Pv) = **{tubes_high}**
- depth rows (air path) = **{int(rows_depth)}**
- Total straight tube pieces used for area = rows_depth × tubes_across × tubes_high
"""
    )

    run_btn = st.button("Run Design & Check", type="primary")


with tabs[1]:
    st.subheader("Results")

    if "results" not in st.session_state:
        st.session_state["results"] = None

    if not st.session_state.get("results") and not st.session_state.get("results") and not st.session_state.get("results") and not st.session_state.get("results"):
        pass  # no-op, keeps lint calm

    if not st.session_state.get("results") and not run_btn:
        st.info("Set inputs in the first tab and click **Run Design & Check**.")
    elif run_btn:
        # --- Fluid properties
        T_mean = 0.5 * (Tw_in + Tw_out)
        proc_fluid = make_process_fluid(fluid_choice, glycol_pct)
        cp, rho, mu, k_fluid = fluid_props(proc_fluid, T_mean, P_kpa)

        # --- Air
        w_in = humidity_ratio_from_tdb_twb(Tdb, Twb, P_kpa)
        rho_air = air_density_kg_per_m3(Tdb, w_in, P_kpa)

        if air_mode == "Estimate from Δh (kJ/kg_da)":
            m_air_est = Q_kw / max(dh_assumed, 1e-9)
            Vdot_air_calc = (m_air_est / max(rho_air, 1e-9)) * 3600.0
        else:
            Vdot_air_calc = Vdot_air

        # --- Merkel area required
        A_req, m_w_auto, m_air, w_in_calc, rho_air_calc, df_profile = merkel_required_area(
            Q_kw=Q_kw,
            Tw_in=Tw_in,
            Tw_out_target=Tw_out,
            Tdb_in=Tdb,
            Twb_in=Twb,
            Vdot_air_m3_h=float(Vdot_air_calc),
            K_kg_s_m2=float(K),
            cp_kj_kgK=float(cp),
            P_kpa=float(P_kpa),
            dA_step_m2=float(dA_step),
            max_area_m2=float(max_area)
        )

        # --- Water flow
        if flow_mode == "User flow (m³/h)":
            flow_m3_h = float(user_flow)
            m_w = (flow_m3_h * rho) / 3600.0
            Q_check = m_w * cp * (Tw_in - Tw_out)
        else:
            m_w = m_w_auto
            flow_m3_h = (m_w * 3600.0) / max(rho, 1e-9)
            Q_check = Q_kw

        # --- Tube counts and area provided
        tubes_across_calc = max(1, int(math.floor(face_W_m / max(Pt_mm/1000.0, 1e-9))))
        tubes_high_calc = max(1, int(math.floor(face_H_m / max(Pv_mm/1000.0, 1e-9))))
        total_tube_pieces = int(rows_depth) * tubes_across_calc * tubes_high_calc

        Do_m = Do_mm / 1000.0
        L_total = total_tube_pieces * tube_length_m
        A_provided = math.pi * Do_m * L_total

        # --- Tube hydraulics per circuit (very first estimate)
        Di_mm = max(0.5, Do_mm - 2.0 * t_mm)
        Di_m = Di_mm / 1000.0

        L_circuit = L_total / max(int(circuits), 1)

        flow_m3_s = flow_m3_h / 3600.0
        flow_per_circuit_m3_s = flow_m3_s / max(int(circuits), 1)
        A_id = math.pi * (Di_m ** 2) / 4.0
        v_int = flow_per_circuit_m3_s / max(A_id, 1e-12)

        Re = reynolds(rho, v_int, Di_m, mu)
        Pr = prandtl(cp, mu, k_fluid)
        Nu = nusselt_dittus_boelter(Re, Pr, heating=False)
        h_i = Nu * k_fluid / max(Di_m, 1e-12)

        dp_pa = dp_darcy(rho, v_int, Di_m, L_circuit, mu, K_minor=float(K_minor))

        # --- Header velocity sanity
        hdr_in_m = hdr_in_mm / 1000.0
        hdr_out_m = hdr_out_mm / 1000.0
        A_hdr_in = math.pi * hdr_in_m**2 / 4.0
        A_hdr_out = math.pi * hdr_out_m**2 / 4.0
        v_hdr_in = flow_m3_s / max(A_hdr_in, 1e-12)
        v_hdr_out = flow_m3_s / max(A_hdr_out, 1e-12)

        # --- Fan power
        Vdot_air_m3_s = float(Vdot_air_calc) / 3600.0
        fan_kw = (Vdot_air_m3_s * float(dP_fan)) / max(float(eta_fan), 1e-9) / 1000.0

        margin_pct = (A_provided / max(A_req, 1e-9) - 1.0) * 100.0

        st.session_state["results"] = {
            "proc_fluid": proc_fluid,
            "cp": cp,
            "rho": rho,
            "mu": mu,
            "k_fluid": k_fluid,
            "A_req": A_req,
            "A_provided": A_provided,
            "margin_pct": margin_pct,
            "flow_m3_h": flow_m3_h,
            "m_w": m_w,
            "m_air": m_air,
            "rho_air": rho_air_calc,
            "w_in": w_in_calc,
            "total_tube_pieces": total_tube_pieces,
            "L_total": L_total,
            "Di_mm": Di_mm,
            "v_int": v_int,
            "Re": Re,
            "Pr": Pr,
            "Nu": Nu,
            "h_i": h_i,
            "dp_kpa": dp_pa / 1000.0,
            "v_hdr_in": v_hdr_in,
            "v_hdr_out": v_hdr_out,
            "fan_kw": fan_kw,
            "Q_check": Q_check,
            "df_profile": df_profile,
            "drawing_png": drawing_png,
            "circuit_df": compute_circuit_distribution(total_tube_pieces, circuits),
            "inputs_snapshot": {
                "Q_kw": Q_kw, "Tw_in": Tw_in, "Tw_out": Tw_out,
                "Tdb": Tdb, "Twb": Twb, "P_kpa": P_kpa,
                "Vdot_air_m3_h": float(Vdot_air_calc),
                "K": K, "dA_step": dA_step, "max_area": max_area,
                "Do_mm": Do_mm, "t_mm": t_mm,
                "rows_depth": int(rows_depth), "Pt_mm": Pt_mm, "Pv_mm": Pv_mm, "Pr_mm": Pr_mm,
                "face_W_m": face_W_m, "face_H_m": face_H_m,
                "tube_length_m": tube_length_m, "circuits": int(circuits),
                "hdr_in_mm": hdr_in_mm, "hdr_out_mm": hdr_out_mm, "K_minor": K_minor,
                "dP_fan": dP_fan, "eta_fan": eta_fan,
                "casing_L_m": casing_L_m, "casing_W_m": casing_W_m, "casing_H_m": casing_H_m,
                "fan_diam_m": fan_diam_m, "eliminator_thk_m": eliminator_thk_m, "sump_depth_m": sump_depth_m,
                "nozzle_count": int(nozzle_count),
            }
        }

    res = st.session_state.get("results")
    if res:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duty (kW)", f"{Q_kw:.2f}")
        c2.metric("Required wetted area (m²)", f"{res['A_req']:.1f}")
        c3.metric("Provided coil area (m²)", f"{res['A_provided']:.1f}")
        c4.metric("Area margin (%)", f"{res['margin_pct']:+.1f}")

        st.write("### Tube bundle and hydraulics (first estimate)")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total tube pieces", f"{res['total_tube_pieces']}")
        d2.metric("Total tube length (m)", f"{res['L_total']:.1f}")
        d3.metric("ΔP per circuit (kPa)", f"{res['dp_kpa']:.2f}")
        d4.metric("Fan power (kW)", f"{res['fan_kw']:.2f}")

        with st.expander("Debug: marching profile (last 50 rows)"):
            st.dataframe(res["df_profile"].tail(50), use_container_width=True)

        with st.expander("Debug: circuit distribution"):
            st.dataframe(res["circuit_df"], use_container_width=True)


with tabs[2]:
    st.subheader("PDF Report")

    res = st.session_state.get("results")
    if not res:
        st.info("Run the calculation first (Results tab).")
    else:
        include_profile = st.checkbox("Include Merkel marching table in PDF", value=True)
        include_drawing = st.checkbox("Include GA + coil drawing in PDF", value=True)
        profile_rows = st.slider("Marching rows to include", min_value=20, max_value=500, value=120, step=20)

        s = res["inputs_snapshot"]

        inputs = {
            "Heat rejection (kW)": f"{s['Q_kw']:.3f}",
            "Process fluid": res["proc_fluid"],
            "Tw in (°C)": s["Tw_in"],
            "Tw out (°C)": s["Tw_out"],
            "Air DB (°C)": s["Tdb"],
            "Air WB (°C)": s["Twb"],
            "Pressure (kPa)": s["P_kpa"],
            "Airflow (m³/h)": f"{s['Vdot_air_m3_h']:.0f}",
            "K (kg/s·m²)": s["K"],
            "Tube OD Do (mm)": s["Do_mm"],
            "Rows depth": s["rows_depth"],
            "Pt (mm)": s["Pt_mm"],
            "Pv (mm)": s["Pv_mm"],
            "Pr (mm)": s["Pr_mm"],
            "Face W (m)": s["face_W_m"],
            "Face H (m)": s["face_H_m"],
            "Tube length per piece (m)": s["tube_length_m"],
            "Circuits": s["circuits"],
            "Header in/out (mm)": f"{s['hdr_in_mm']}/{s['hdr_out_mm']}",
            "Fan ΔP (Pa)": s["dP_fan"],
            "Fan η": s["eta_fan"],
            "GA L×W×H (m)": f"{s['casing_L_m']:.2f}×{s['casing_W_m']:.2f}×{s['casing_H_m']:.2f}",
            "Fan diameter (m)": s["fan_diam_m"],
            "Nozzles (label)": s["nozzle_count"],
        }

        outputs = {
            "Required coil area (m²)": f"{res['A_req']:.3f}",
            "Provided coil area (m²)": f"{res['A_provided']:.3f}",
            "Area margin (%)": f"{res['margin_pct']:.2f}",
            "Process flow (m³/h)": f"{res['flow_m3_h']:.3f}",
            "Fan power (kW)": f"{res['fan_kw']:.3f}",
            "Tube ΔP per circuit (kPa)": f"{res['dp_kpa']:.3f}",
            "Energy check Q_check (kW)": f"{res['Q_check']:.3f}",
        }

        intermediates = {
            "cp (kJ/kg·K)": f"{res['cp']:.6f}",
            "ρ (kg/m³)": f"{res['rho']:.3f}",
            "μ (mPa·s)": f"{res['mu']*1000.0:.6f}",
            "k (W/m·K)": f"{res['k_fluid']:.6f}",
            "m_w (kg/s)": f"{res['m_w']:.6f}",
            "m_air (kg/s)": f"{res['m_air']:.6f}",
            "w_in (kg/kg_da)": f"{res['w_in']:.6f}",
            "ρ_air (kg/m³)": f"{res['rho_air']:.6f}",
            "Tube ID (mm)": f"{res['Di_mm']:.3f}",
            "Internal v (m/s)": f"{res['v_int']:.4f}",
            "Re": f"{res['Re']:.0f}",
            "Pr": f"{res['Pr']:.4f}",
            "Nu": f"{res['Nu']:.2f}",
            "h_i (W/m²·K)": f"{res['h_i']:.1f}",
            "Header v_in (m/s)": f"{res['v_hdr_in']:.3f}",
            "Header v_out (m/s)": f"{res['v_hdr_out']:.3f}",
        }

        layout_summary = {
            "Total tube pieces": res["total_tube_pieces"],
            "Total tube length (m)": f"{res['L_total']:.2f}",
            "Rows depth": s["rows_depth"],
            "Face W×H (m)": f"{s['face_W_m']:.3f}×{s['face_H_m']:.3f}",
            "Pitches Pt/Pv/Pr (mm)": f"{s['Pt_mm']}/{s['Pv_mm']}/{s['Pr_mm']}",
            "Tube Do/thk (mm)": f"{s['Do_mm']}/{s['t_mm']}",
            "Circuits": s["circuits"],
        }

        pdf_bytes = build_pdf_report(
            title="Evaporative Fluid Cooler Coil Design Report",
            inputs=inputs,
            outputs=outputs,
            intermediates=intermediates,
            layout_summary=layout_summary,
            circuit_df=res["circuit_df"],
            df_profile=res["df_profile"],
            drawing_png=res.get("drawing_png"),
            include_profile=include_profile,
            profile_rows=profile_rows,
            include_drawing=include_drawing,
        )

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="evap_fluid_cooler_report.pdf",
            mime="application/pdf"
        )
