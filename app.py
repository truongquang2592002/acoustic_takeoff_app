"""
Acoustic Panel Takeoff, 2D Nesting & Quality Control System
Multi-Page Batch Processing & Master BOQ Generation
An enterprise AEC Streamlit application for automated multi-page vector PDF takeoff,
independent global material nesting optimization, contractor drafting QC audit,
and professional Master BOQ Excel workbook generation.
"""

from datetime import datetime
import io
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure root workspace is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Import core modules
from modules import (
    GlobalNestingResult,
    NestedPiece,
    NestedSheet,
    NestingResult,
    PDFPageData,
    PanelItem,
    QCIssue,
    QCReport,
    VectorPath,
    compute_scale_multiplier,
    detect_drawing_scales,
    export_takeoff_to_excel,
    extract_pdf_data,
    generate_sample_contractor_drawing,
    process_geometry_takeoff,
    render_issue_zoom_crop,
    render_nested_sheet_diagram,
    render_qc_overlay_diagram,
    run_nesting_optimization,
    run_qc_validation,
    scan_full_document
)

# Page Configuration
st.set_page_config(
    page_title="Acoustic Takeoff & Master BOQ Engine",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Slate/Navy AEC Aesthetic)
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #CBD5E1;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #0284C7;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
        font-size: 0.82rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0F172A;
        font-weight: 700;
    }
    /* Headers */
    h1, h2, h3 {
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    /* Badges */
    .material-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-poly { background-color: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }
    .badge-wood { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .badge-fabric { background-color: #D1FAE5; color: #047857; border: 1px solid #A7F3D0; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F8FAFC;
        border-radius: 8px 8px 0 0;
        gap: 6px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E2E8F0;
        font-weight: bold;
        color: #0284C7;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "panels" not in st.session_state:
    st.session_state.panels = []
if "cutouts" not in st.session_state:
    st.session_state.cutouts = []
if "nesting_result" not in st.session_state:
    st.session_state.nesting_result = None
if "qc_report" not in st.session_state:
    st.session_state.qc_report = None
if "current_inspect_page" not in st.session_state:
    st.session_state.current_inspect_page = 0
if "single_page_data" not in st.session_state:
    st.session_state.single_page_data = None
if "qc_overlay_img" not in st.session_state:
    st.session_state.qc_overlay_img = None
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0


# Helper to load sample Multi-Page PDF
def load_sample_multi_page_pdf(with_drafting_errors: bool = True):
    sample_path = BASE_DIR / "data" / "sample_acoustic_dwg.pdf"
    generate_sample_contractor_drawing(sample_path, include_drafting_errors=with_drafting_errors, num_pages=3)
    with open(sample_path, "rb") as f:
        data_bytes = f.read()
    st.session_state.pdf_bytes = data_bytes
    st.session_state.pdf_filename = "Orion_Hub_3Page_Fitout_Package.pdf"
    st.session_state.scan_results = None
    st.session_state.panels = []
    st.session_state.cutouts = []
    st.session_state.nesting_result = None
    st.session_state.qc_report = None
    st.session_state.current_inspect_page = 0
    st.session_state.single_page_data = extract_pdf_data(data_bytes, page_number=0, render_dpi=130)
    st.session_state.total_pages = st.session_state.single_page_data.total_pages


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title("📐 Acoustic Takeoff & BOQ")
    st.caption("Multi-Page Batch Scanner & Global Material Nesting")
    st.divider()

    # 1. PDF Upload Section
    st.subheader("1. Drawing Document Source")
    uploaded_file = st.file_uploader(
        "Upload Multi-Page Drawing (PDF)",
        type=["pdf"],
        help="Upload a multi-page architectural set containing wall/ceiling acoustic linework."
    )

    st.markdown("**Load Demo Multi-Page Sets:**")
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("⚡ 3-Page Set (With QC)", use_container_width=True, help="Load Orion Hub 3-page set (Boardroom, Cover Sheet, Auditorium) with drafting flaws"):
            load_sample_multi_page_pdf(with_drafting_errors=True)
            st.rerun()

    with col_btn2:
        if st.button("✨ 3-Page Clean Set", use_container_width=True, help="Load pristine 3-page set without drafting flaws"):
            load_sample_multi_page_pdf(with_drafting_errors=False)
            st.rerun()

    if st.button("🔄 Reset Project", use_container_width=True):
        st.session_state.pdf_bytes = None
        st.session_state.pdf_filename = None
        st.session_state.scan_results = None
        st.session_state.panels = []
        st.session_state.cutouts = []
        st.session_state.nesting_result = None
        st.session_state.qc_report = None
        st.session_state.single_page_data = None
        st.session_state.total_pages = 0
        st.rerun()

    if uploaded_file is not None and (st.session_state.pdf_filename != uploaded_file.name):
        pdf_bytes = uploaded_file.read()
        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.pdf_filename = uploaded_file.name
        st.session_state.scan_results = None
        st.session_state.panels = []
        st.session_state.cutouts = []
        st.session_state.nesting_result = None
        st.session_state.qc_report = None
        st.session_state.single_page_data = extract_pdf_data(pdf_bytes, page_number=0, render_dpi=130)
        st.session_state.total_pages = st.session_state.single_page_data.total_pages

    st.divider()

    # 2. Scale & Units Calibration
    st.subheader("2. Project Scale & Units")
    target_unit = st.selectbox("Measurement Units", options=["mm", "m", "inch", "ft"], index=0)

    preset_scales = [
        "1:50", "1:100", "1:20", "1:25", "1:10", "1:200", "1:1 (Full Scale)",
        '1/4" = 1\'-0" (1:48)', '1/8" = 1\'-0" (1:96)', '1/2" = 1\'-0" (1:24)',
        "Custom Calibration"
    ]

    default_scale_idx = 0
    if st.session_state.single_page_data and st.session_state.single_page_data.detected_scales:
        detected = st.session_state.single_page_data.detected_scales[0]["label"]
        st.success(f"🎯 Detected on Sheet 1: **{detected}**")
        if detected in preset_scales:
            default_scale_idx = preset_scales.index(detected)

    chosen_scale = st.selectbox("Drawing Scale Preset", options=preset_scales, index=default_scale_idx)

    custom_pts = 72.0
    custom_units = 1000.0
    if chosen_scale == "Custom Calibration":
        st.caption("Specify known measurement calibration:")
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            custom_pts = st.number_input("Drawing Points (pt)", min_value=1.0, value=56.7, step=1.0)
        with c_col2:
            custom_units = st.number_input(f"Real Distance ({target_unit})", min_value=1.0, value=1000.0, step=10.0)

    scale_multiplier = compute_scale_multiplier(
        scale_preset=chosen_scale,
        target_unit=target_unit,
        custom_drawing_pts=custom_pts,
        custom_real_units=custom_units
    )
    st.caption(f"Scale Multiplier: 1 pt = **{scale_multiplier:.4f} {target_unit}**")

    st.divider()

    # 3. Stock Sheet Specification
    st.subheader("3. Stock Sheet Specifications")
    sheet_presets = {
        "2400 × 1200 mm (Standard Metric Board)": (2400.0, 1200.0),
        "2440 × 1220 mm (8ft × 4ft Board)": (2440.0, 1220.0),
        "3000 × 1200 mm (10ft × 4ft Large Sheet)": (3000.0, 1200.0),
        "1200 × 600 mm (Acoustic Ceiling Tile)": (1200.0, 600.0),
        "48 × 96 inch (4x8 ft Imperial)": (2438.4, 1219.2) if target_unit == "mm" else (96.0, 48.0),
        "Custom Dimensions": (2400.0, 1200.0)
    }

    selected_sheet_preset = st.selectbox("Stock Preset", options=list(sheet_presets.keys()))
    default_sw, default_sh = sheet_presets[selected_sheet_preset]

    col_sw, col_sh = st.columns(2)
    with col_sw:
        stock_w = st.number_input(f"Stock Length ({target_unit})", value=float(default_sw), step=50.0)
    with col_sh:
        stock_h = st.number_input(f"Stock Width ({target_unit})", value=float(default_sh), step=50.0)

    col_k, col_m = st.columns(2)
    with col_k:
        blade_kerf = st.number_input(f"Blade Kerf ({target_unit})", value=3.0 if target_unit == "mm" else 0.125, step=0.5)
    with col_m:
        sheet_margin = st.number_input(f"Sheet Margin ({target_unit})", value=10.0 if target_unit == "mm" else 0.5, step=1.0)

    allow_rot = st.checkbox("Allow 90° Piece Rotation", value=True, help="Disable if materials have directional grain or pattern.")
    nest_algo = st.selectbox(
        "Packing Algorithm",
        options=["MaxRectsBssf", "MaxRectsBl", "SkylineBl", "GuillotineBssfSas"],
        index=0
    )

    st.divider()

    # 4. Material Cost Model
    st.subheader("4. Material Cost Rates")
    cost_poly = st.number_input("Polyester 9mm Sheet ($)", value=65.0, step=5.0)
    cost_wood = st.number_input("Wood Slat Panel Sheet ($)", value=115.0, step=5.0)
    cost_fabric = st.number_input("Fabric Wrapped 25mm Sheet ($)", value=88.0, step=5.0)
    cost_other = st.number_input("Standard/Other Sheet ($)", value=60.0, step=5.0)

    material_cost_rates = {
        "Polyester 9mm": cost_poly,
        "Wood Slat Panel": cost_wood,
        "Fabric Wrapped 25mm": cost_fabric,
        "Acoustic Plasterboard": cost_other
    }


# ==========================================
# MAIN APP BODY
# ==========================================

st.title("🏗️ Multi-Page Acoustic Takeoff & Master BOQ Engine")
st.markdown("Batch scan multi-page architectural PDFs, auto-classify acoustic materials, deduct page-level cutouts, optimize global 2D nesting by material type, and generate professional Master BOQ Excel reports.")

# Check if document loaded
if st.session_state.pdf_bytes is None:
    st.info("👋 **Welcome to the Acoustic Takeoff Engine!** Upload a multi-page architectural PDF drawing or load one of the built-in demo sets below.")

    col_d1, col_d2, _ = st.columns([1.2, 1.2, 1.6])
    with col_d1:
        if st.button("🚀 Load 3-Page Project with QC Defects", type="primary", use_container_width=True):
            load_sample_multi_page_pdf(with_drafting_errors=True)
            st.rerun()
    with col_d2:
        if st.button("✨ Load 3-Page Pristine Project", use_container_width=True):
            load_sample_multi_page_pdf(with_drafting_errors=False)
            st.rerun()

    st.markdown("""
    ---
    ### 🌟 System Capabilities:
    1. **Multi-Page Batch Scanner**: Automatically inspects all pages, detecting Room Names / Sheet Titles using PyMuPDF word-level analysis.
    2. **Acoustic Material Classification**: Vector linework and fills are classified into **Polyester 9mm**, **Wood Slat Panels**, **Fabric Wrapped 25mm**, or **Cutouts/Voids**.
    3. **Page-by-Page Cutout Deductions**: Deducts door openings, glazing voids, downlights, and HVAC slots local to each page.
    4. **Graceful Empty Page Skipping**: Text-only specification cover sheets or notes pages with no acoustic materials are gracefully skipped.
    5. **Global Material Nesting Optimization**: Panels of the same material from Page 1, 3, etc. are combined and packed together for optimal global board yield.
    6. **Master BOQ Excel Exporter**: Professional 3+ sheet AEC spreadsheet containing an Executive Summary, itemized Detailed BOQ, and material-separated nesting cutting plans.
    """)
    st.stop()


# Document Banner & Full Project Takeoff Trigger
total_pages = st.session_state.total_pages or 1
doc_name = st.session_state.pdf_filename or "Project_Drawings.pdf"

st.markdown(f"**Loaded Document:** `{doc_name}` &nbsp;|&nbsp; **Pages:** `{total_pages}` &nbsp;|&nbsp; **Scale:** `{chosen_scale}` ({scale_multiplier:.3f} {target_unit}/pt)")

# Trigger Full Takeoff Button
run_col1, run_col2 = st.columns([1.4, 2.6])
with run_col1:
    btn_run = st.button("🚀 Run Full Project Takeoff", type="primary", use_container_width=True)

# Run Batch Process when clicked or auto-run on first load if not yet done
if btn_run or (st.session_state.scan_results is None and st.session_state.pdf_bytes is not None):
    with st.status("🚀 Running Full Project Takeoff & Batch Processing...", expanded=True) as status:
        p_bar = st.progress(0.0)

        # Step 1: Scan full document
        st.write(f"📄 Scanning {total_pages} drawing pages to identify acoustic materials & room metadata...")
        scan_data = scan_full_document(
            pdf_path=st.session_state.pdf_bytes,
            scale_factor=scale_multiplier,
            render_previews=True,
            preview_dpi=110
        )
        st.session_state.scan_results = scan_data
        p_bar.progress(0.35)

        # Step 2: Page by page processing notice
        for p_no, p_info in sorted(scan_data["pages"].items()):
            mats = list(p_info["materials"].keys())
            if mats:
                st.write(f"  • **Page {p_no}** [{p_info['sheet_name']}]: Identified **{p_info['total_material_paths']}** acoustic entities ({', '.join(mats)})")
            else:
                st.write(f"  • **Page {p_no}** [{p_info['sheet_name']}]: *No acoustic materials detected (Gracefully Skipped)*")

        p_bar.progress(0.55)

        # Step 3: Geometry & Void Deduction
        st.write("📐 Calculating geometries, oriented bounding boxes, and deducting openings...")
        panels, cutouts = process_geometry_takeoff(
            document_scan_data=scan_data,
            scale_factor=scale_multiplier,
            min_area_m2=0.04,
            max_area_m2=60.0,
            unit=target_unit
        )
        st.session_state.panels = panels
        st.session_state.cutouts = cutouts
        p_bar.progress(0.75)

        # Step 4: Global Material Nesting Optimization
        st.write("📦 Running independent 2D bin packing globally for each material type...")
        nesting_res = run_nesting_optimization(
            panels=panels,
            stock_width=stock_w,
            stock_height=stock_h,
            unit=target_unit,
            blade_kerf=blade_kerf,
            allow_rotation=allow_rot,
            algorithm_name=nest_algo,
            sheet_margin=sheet_margin
        )
        st.session_state.nesting_result = nesting_res
        p_bar.progress(0.90)

        # Step 5: QC Check on Page 1 / active page
        st.write("🛡️ Validating contractor drafting standards & scale fidelity...")
        if st.session_state.single_page_data is not None:
            qc_rep = run_qc_validation(
                pdf_data=st.session_state.single_page_data,
                panels=panels,
                scale_multiplier=scale_multiplier,
                chosen_scale_preset=chosen_scale,
                target_unit=target_unit
            )
            st.session_state.qc_report = qc_rep
            st.session_state.qc_overlay_img = render_qc_overlay_diagram(st.session_state.single_page_data.preview_image, qc_rep, dpi=130)

        p_bar.progress(1.0)
        if status is not None and hasattr(status, "update"):
            status.update(label=f"✅ Full Takeoff Complete! Extracted {len(panels)} panels across project.", state="complete", expanded=False)

    st.rerun()


# Load current results from session state
panels: List[PanelItem] = st.session_state.panels
nesting_result: GlobalNestingResult = st.session_state.nesting_result
scan_data = st.session_state.scan_results

if not panels:
    st.warning("No acoustic panels found in the document. Please check the drawing or scale calibration.")
    st.stop()


# ==========================================
# WORKSPACE TABS
# ==========================================
tab_master, tab_schedule, tab_nesting, tab_qc = st.tabs([
    "📊 1. Master BOQ & Material Summary",
    "📋 2. Itemized Takeoff Schedule",
    "📦 3. Global Nesting by Material",
    "🔍 4. Drawing Inspector & QC Audit"
])


# ------------------------------------------
# TAB 1: MASTER BOQ & MATERIAL SUMMARY
# ------------------------------------------
with tab_master:
    st.subheader("Master Project Summary & Aggregated Material Takeoff")

    # Global KPI Row
    total_net_m2 = sum(p.net_area for p in panels)
    total_panels_count = len(panels)
    total_sheets_req = nesting_result.total_sheets if nesting_result else 0
    overall_yield_pct = nesting_result.overall_yield_pct if nesting_result else 0.0
    overall_waste_pct = nesting_result.overall_waste_pct if nesting_result else 0.0

    total_est_cost = sum(
        res.total_sheets * material_cost_rates.get(m_name, cost_poly)
        for m_name, res in nesting_result.material_results.items()
    ) if (nesting_result and hasattr(nesting_result, "material_results")) else 0.0

    kpi_m1, kpi_m2, kpi_m3, kpi_m4, kpi_m5 = st.columns(5)
    kpi_m1.metric("Total Net Acoustic Area", f"{total_net_m2:,.2f} m²", help="Net billable area after cutout deductions.")
    kpi_m2.metric("Total Panels / Tiles", f"{total_panels_count:,} pcs")
    kpi_m3.metric("Total Stock Sheets Req.", f"{total_sheets_req:,} sheets", f"Size: {int(stock_w)}×{int(stock_h)} {target_unit}")
    kpi_m4.metric("Overall Material Yield", f"{overall_yield_pct:.1f}%", f"{overall_waste_pct:.1f}% offcut scrap")
    kpi_m5.metric("Total Estimated Cost", f"${total_est_cost:,.2f}", "Across all materials")

    st.markdown("---")

    # MASTER DATA TABLE (GROUPED BY MATERIAL TYPE)
    st.markdown("##### 📦 Master Material Summary Table (Aggregated Project Scope)")
    st.caption("Aggregated totals per acoustic material type across all pages. 2D nesting run independently per material for global sheet yield optimization.")

    mat_table_rows = []
    all_mat_names = sorted(list(set(p.material_type for p in panels)))
    for mat_name in all_mat_names:
        mat_panels = [p for p in panels if p.material_type == mat_name]
        mat_net_m2 = sum(p.net_area for p in mat_panels)
        res = nesting_result.material_results.get(mat_name) if (nesting_result and hasattr(nesting_result, "material_results")) else None
        sheets_req = res.total_sheets if res else math.ceil(mat_net_m2 / 2.88)
        stk_area = res.total_stock_area_m2 if res else sheets_req * 2.88
        yield_val = f"{res.overall_yield_pct:.1f}%" if res else f"{(mat_net_m2 / max(0.001, stk_area))*100:.1f}%"
        waste_val = f"{res.overall_waste_pct:.1f}%" if res else f"{max(0.0, 100.0 - (mat_net_m2 / max(0.001, stk_area))*100):.1f}%"
        rate = material_cost_rates.get(mat_name, cost_poly)
        mat_cost = sheets_req * rate

        mat_table_rows.append({
            "Material Type": mat_name,
            "Panel Count": len(mat_panels),
            "Total Net Area (m²)": round(mat_net_m2, 2),
            "Stock Sheets Req.": sheets_req,
            "Total Stock Area (m²)": round(stk_area, 2),
            "Yield %": yield_val,
            "Waste %": waste_val,
            "Sheet Rate ($)": f"${rate:,.2f}",
            "Est. Material Cost ($)": f"${mat_cost:,.2f}"
        })

    df_master_materials = pd.DataFrame(mat_table_rows)
    st.dataframe(df_master_materials, hide_index=True, use_container_width=True)

    # Master Excel Download Button
    excel_bytes = export_takeoff_to_excel(
        panels=panels,
        nesting=nesting_result,
        qc_report=st.session_state.qc_report,
        project_name=doc_name.replace(".pdf", ""),
        drawing_ref=f"Drawing Set ({total_pages} Pages)",
        scale_label=chosen_scale,
        cost_per_sheet=cost_poly,
        material_cost_rates=material_cost_rates,
        unit=target_unit
    )

    col_dl1, col_dl2 = st.columns([1.5, 2.5])
    with col_dl1:
        st.download_button(
            label="📥 Download Master BOQ Report (.xlsx)",
            data=excel_bytes,
            file_name=f"{doc_name.replace('.pdf', '')}_Master_BOQ_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )
    with col_dl2:
        st.caption("Excel includes Sheet 1 (Master Summary), Sheet 2 (Detailed BOQ with Page & Room Name), and Sheet 3 (Nesting Layouts by Material).")

    st.markdown("---")

    # PAGE-BY-PAGE SUMMARY BREAKDOWN
    st.markdown("##### 📄 Page-by-Page Scope Breakdown")
    page_rows = []
    page_dict = scan_data["pages"] if scan_data else {}

    for p_no in sorted(page_dict.keys()):
        p_info = page_dict[p_no]
        p_panels = [p for p in panels if p.page_number == p_no]
        mats_present = ", ".join(sorted(list(set(p.material_type for p in p_panels)))) if p_panels else "None (Cover/Notes)"
        p_net = sum(p.net_area for p in p_panels)
        p_gross = sum(p.gross_area for p in p_panels)
        p_cut = sum(p.cutout_area for p in p_panels)
        p_pct = (p_net / max(0.001, total_net_m2)) * 100.0

        page_rows.append({
            "Page No.": f"Page {p_no}",
            "Drawing / Room Name": p_info.get("sheet_name", f"Sheet {p_no}"),
            "Materials Present": mats_present,
            "Panel Count": len(p_panels),
            "Gross Area (m²)": round(p_gross, 2),
            "Cutout Area (m²)": round(p_cut, 2),
            "Net Area (m²)": round(p_net, 2),
            "% of Project Net Area": f"{p_pct:.1f}%"
        })

    df_page_summary = pd.DataFrame(page_rows)
    st.dataframe(df_page_summary, hide_index=True, use_container_width=True)


# ------------------------------------------
# TAB 2: ITEMIZED TAKEOFF SCHEDULE
# ------------------------------------------
with tab_schedule:
    st.subheader("Itemized Takeoff Schedule (Detailed BOQ)")
    st.caption("Complete bill of quantities itemizing every acoustic panel with its originating page, room, dimensions, cutouts, and net area.")

    # Filter Controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_mat_filter = st.selectbox(
            "Filter by Material Type:",
            options=["All Materials"] + sorted(list(set(p.material_type for p in panels)))
        )
    with col_f2:
        page_options = ["All Pages"] + [f"Page {p_no}" for p_no in sorted(list(set(p.page_number for p in panels)))]
        selected_page_filter = st.selectbox("Filter by Drawing Page:", options=page_options)

    # Filter data
    filtered_panels = panels
    if selected_mat_filter != "All Materials":
        filtered_panels = [p for p in filtered_panels if p.material_type == selected_mat_filter]
    if selected_page_filter != "All Pages":
        target_p = int(selected_page_filter.replace("Page ", ""))
        filtered_panels = [p for p in filtered_panels if p.page_number == target_p]

    schedule_records = []
    for p in filtered_panels:
        schedule_records.append({
            "Page No.": f"Page {p.page_number}",
            "Drawing/Room Name": p.sheet_name,
            "Material Type": p.material_type,
            "Panel ID": p.id,
            "Item Tag": p.tag,
            "L × W": p.dimension_str,
            f"Cut Length ({target_unit})": p.cut_length,
            f"Cut Width ({target_unit})": p.cut_width,
            "Cutouts": f"{p.cutout_count} void(s)" if p.cutout_count > 0 else "0",
            "Net Area (m²)": p.net_area,
            "Gross Area (m²)": p.gross_area,
            f"Perimeter ({target_unit})": p.perimeter,
            "Category": p.category,
            "CAD Layer": p.layer or "Default"
        })

    df_schedule = pd.DataFrame(schedule_records)
    st.dataframe(df_schedule, use_container_width=True, height=450)
    st.caption(f"Showing **{len(filtered_panels)}** of **{len(panels)}** project items.")


# ------------------------------------------
# TAB 3: GLOBAL NESTING BY MATERIAL
# ------------------------------------------
with tab_nesting:
    st.subheader("2D Stock Sheet Nesting (Independently Grouped by Material)")
    st.caption("Panels of each material type from all pages are nested together onto standard stock sheets to achieve maximum global yield.")

    available_materials = sorted(list(nesting_result.material_results.keys())) if (nesting_result and hasattr(nesting_result, "material_results")) else []

    if not available_materials:
        st.info("No nesting results available.")
    else:
        chosen_mat = st.radio(
            "Select Material to Inspect Nesting Plans:",
            options=available_materials,
            horizontal=True
        )

        mat_nest: NestingResult = nesting_result.material_results[chosen_mat]

        # Material Top KPIs
        n_col1, n_col2, n_col3, n_col4 = st.columns(4)
        n_col1.metric("Stock Sheets Required", f"{mat_nest.total_sheets} sheets", f"Material: {chosen_mat}")
        n_col2.metric("Material Yield %", f"{mat_nest.overall_yield_pct:.1f}%", help="Usable panel area vs stock sheet area purchased.")
        n_col3.metric("Scrap & Offcut Waste", f"{mat_nest.overall_waste_pct:.1f}%", f"{mat_nest.total_waste_area_m2:.2f} m² offcut")
        n_col4.metric("Placed Pieces", f"{mat_nest.total_placed_pieces} pcs", f"{len(mat_nest.unplaced_items)} unplaced")

        st.markdown("---")

        # Cutting Diagram Viewer for Chosen Material
        st.markdown(f"##### Stock Sheet Cutting Layout Diagrams — {chosen_mat}")

        if mat_nest.sheets:
            sheet_choices = [
                f"Sheet #{s.sheet_index} — {len(s.pieces)} pieces | Yield {s.yield_pct}%"
                for s in mat_nest.sheets
            ]
            selected_sheet_str = st.selectbox("Select Sheet Diagram:", options=sheet_choices)
            selected_s_idx = int(selected_sheet_str.split("#")[1].split(" ")[0]) - 1
            active_sheet = mat_nest.sheets[selected_s_idx]

            col_diag, col_details = st.columns([2.2, 1.0])

            with col_diag:
                with st.spinner("Rendering engineering cutting diagram..."):
                    sheet_img = render_nested_sheet_diagram(active_sheet, dpi=130)

                if sheet_img:
                    st.image(sheet_img, use_container_width=True, caption=f"Stock Sheet #{active_sheet.sheet_index} [{chosen_mat}] Cut Plan")

                    # Download diagram PNG
                    img_buf = io.BytesIO()
                    sheet_img.save(img_buf, format="PNG")
                    st.download_button(
                        label=f"📥 Download Sheet #{active_sheet.sheet_index} Diagram (PNG)",
                        data=img_buf.getvalue(),
                        file_name=f"{chosen_mat.replace(' ', '_')}_sheet_{active_sheet.sheet_index}.png",
                        mime="image/png"
                    )

            with col_details:
                st.markdown(f"**Sheet #{active_sheet.sheet_index} Cutting Schedule**")
                piece_rows = []
                for p in active_sheet.pieces:
                    piece_rows.append({
                        "Tag": p.tag,
                        "Page": f"P.{p.page_number}",
                        f"W ({target_unit})": int(round(p.width)),
                        f"H ({target_unit})": int(round(p.height)),
                        "Rotated": "Yes" if p.is_rotated else "No"
                    })
                st.dataframe(pd.DataFrame(piece_rows), hide_index=True, use_container_width=True)

                st.markdown("**Remnant Offcuts**")
                rem_rows = []
                for r in active_sheet.remnants:
                    rem_rows.append({
                        f"Dim ({target_unit})": f"{int(round(r.width))}×{int(round(r.height))}",
                        "Status": "Usable" if r.is_usable else "Scrap"
                    })
                if rem_rows:
                    st.dataframe(pd.DataFrame(rem_rows), hide_index=True, use_container_width=True)
                else:
                    st.caption("No significant remnants on this sheet.")
        else:
            st.info("No sheets generated for this material.")


# ------------------------------------------
# TAB 4: DRAWING INSPECTOR & QC AUDIT
# ------------------------------------------
with tab_qc:
    st.subheader("Drawing Page Inspector & Quality Control Audit")
    st.caption("Inspect individual page linework, preview vector geometry, and audit contractor drafting defects (unclosed loops, fake dimensions).")

    # Select which page to inspect
    inspect_page_idx = st.selectbox(
        "Select Drawing Page to Inspect:",
        options=list(range(total_pages)),
        format_func=lambda idx: f"Page {idx + 1} of {total_pages} — {scan_data['pages'][idx + 1]['sheet_name'] if scan_data and (idx + 1) in scan_data['pages'] else f'Page {idx+1}'}"
    )

    # Load page data if different from current
    if st.session_state.current_inspect_page != inspect_page_idx:
        st.session_state.current_inspect_page = inspect_page_idx
        with st.spinner(f"Loading Page {inspect_page_idx + 1}..."):
            st.session_state.single_page_data = extract_pdf_data(st.session_state.pdf_bytes, page_number=inspect_page_idx, render_dpi=130)
            st.session_state.qc_report = run_qc_validation(
                pdf_data=st.session_state.single_page_data,
                panels=[p for p in panels if p.page_number == (inspect_page_idx + 1)],
                scale_multiplier=scale_multiplier,
                chosen_scale_preset=chosen_scale,
                target_unit=target_unit
            )
            st.session_state.qc_overlay_img = render_qc_overlay_diagram(st.session_state.single_page_data.preview_image, st.session_state.qc_report, dpi=130)
        st.rerun()

    active_page_data: PDFPageData = st.session_state.single_page_data
    qc_rep: Optional[QCReport] = st.session_state.qc_report

    # Top QC Metrics for this page
    if qc_rep:
        q_c1, q_c2, q_c3, q_c4 = st.columns(4)
        q_c1.metric("Drafting Health Score", f"{qc_rep.health_score:.1f}%", f"Grade: {qc_rep.health_grade}")
        q_c2.metric("Critical Defects", f"{qc_rep.critical_count} Issues", "Double counting / unclosed" if qc_rep.critical_count > 0 else "Clean")
        q_c3.metric("Warnings", f"{qc_rep.warning_count} Items")
        q_c4.metric("Scale Sanity", f"{qc_rep.scale_validation_status}")

    st.markdown("---")

    col_preview, col_defects = st.columns([1.8, 1.2])

    with col_preview:
        view_opt = st.radio("View Mode:", options=["🛡️ Annotated QC Defect Overlay", "📄 Original PDF Linework"], horizontal=True)

        if view_opt == "🛡️ Annotated QC Defect Overlay" and st.session_state.qc_overlay_img is not None:
            st.image(st.session_state.qc_overlay_img, use_container_width=True, caption=f"Page {inspect_page_idx + 1} with QC Defect Callouts")
        else:
            if active_page_data and active_page_data.preview_image:
                st.image(active_page_data.preview_image, use_container_width=True, caption=f"Page {inspect_page_idx + 1} Original Linework")

    with col_defects:
        st.markdown("##### 🔍 Interactive Defect Spotlight")
        if qc_rep and qc_rep.issues:
            issue_titles = [f"[{i.severity}] {i.id}: {i.category} - {i.title[:35]}" for i in qc_rep.issues]
            sel_issue_idx = st.selectbox(
                "Select Issue to Inspect Up-Close:",
                options=list(range(len(qc_rep.issues))),
                format_func=lambda idx: issue_titles[idx]
            )
            curr_issue: QCIssue = qc_rep.issues[sel_issue_idx]

            if active_page_data and active_page_data.preview_image:
                crop = render_issue_zoom_crop(active_page_data.preview_image, curr_issue, padding_pt=45.0)
                if crop:
                    st.image(crop, use_container_width=True, caption=f"Zoom: {curr_issue.id} ({curr_issue.category})")

            st.markdown(f"**Issue Description:**")
            st.info(curr_issue.description)

            if curr_issue.deviation_pct:
                st.warning(f"⚠️ Measured Deviation: **{curr_issue.deviation_pct:.1f}%**")

            st.markdown(f"**💡 Remediation:**")
            st.success(curr_issue.remedy)
        else:
            st.success("🎉 No drafting defects on this page! Clean linework verified.")


# Footer
st.markdown("---")
st.caption("Antigravity AEC Takeoff Engine | Multi-Page Batch Scanner, Global Material Nesting & Master BOQ | Production Ready")
