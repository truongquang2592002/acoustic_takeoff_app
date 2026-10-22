# Acoustic Panel Takeoff, Multi-Page Batch Scanner & Master BOQ Engine

A complete, enterprise AEC Python tool that automates **Multi-Page Architectural Acoustic Takeoff**, **Material Classification**, **Independent Global 2D Nesting**, and **Master BOQ Generation** with contractor drafting QC validation.

---

## 🏗️ Core Architecture & Features

1. **Multi-Page Batch Scanner & Material Classifier (`1_pdf_parser.py`)**:
   - `scan_full_document(pdf_path, scale_factor)`: Iterates across all pages of large, multi-page architectural PDFs.
   - Word-level text analysis (`page.get_text("words")`) to auto-detect Sheet Names, Room Names, and Title Blocks.
   - Classifies acoustic materials based on vector line colors and fills:
     - 🟦 **Polyester 9mm** (Cyan / Blue vectors)
     - 🟧 **Wood Slat Panels** (Amber / Orange vectors)
     - 🟩 **Fabric Wrapped 25mm** (Emerald / Green vectors)
     - 🟥 **Cutouts / Voids** (Red dashed linework)
   - Built-in multi-page demo generator (`data/sample_acoustic_dwg.pdf`) featuring Boardroom, Cover Sheet, and Auditorium.

2. **Global Geometry Takeoff & Void Deduction (`2_geometry.py`)**:
   - Accepts multi-page nested scan structures or single-page vector paths.
   - Deducts door/window openings, lighting cutouts, and HVAC voids page-by-page.
   - Gracefully skips pages without acoustic materials (e.g. general notes or cover pages).
   - Computes Oriented Bounding Boxes (OBB), fabrication dimensions ($L \times W$), gross areas, net areas, and perimeters.
   - Retains `page_number`, `sheet_name`, and `material_type` for every itemized panel.

3. **Global Material Nesting Optimization (`3_nesting.py`)**:
   - `run_nesting_optimization(panels)` runs **independently for each material type across the whole project**.
   - Nests all "Polyester 9mm" panels from Pages 1, 3, etc. together to maximize global board yield.
   - 2D bin packing using `rectpack` (MaxRects / Skyline) with saw kerf and grain-rotation controls.
   - Generates material-separated cutting plans and remnant offcut inventories ($> 500\text{ mm}$).

4. **Master BOQ Excel Exporter (`4_exporter.py`)**:
   - Generates executive-ready, AEC-standard `.xlsx` workbooks:
     - **Sheet 1 (Master Summary):** Global KPIs, project metadata, and aggregated material table (`Piece Count`, `Net Area m²`, `Stock Sheets Req.`, `Yield %`, `Waste %`, `Est. Cost`).
     - **Sheet 2 (Detailed BOQ):** Itemized schedule with `Page No.`, `Drawing/Room Name`, `Material Type`, `Panel ID`, `L x W`, `Cutouts`, `Net Area`, `Gross Area`, and `Perimeter`.
     - **Sheet 3 (Nesting by Material):** Sheet-by-sheet cutting plans, yield analytics, and CNC saw coordinates separated by material.

5. **Streamlit Batch Processing UI (`app.py`)**:
   - **Batch Processing Mode** with real-time `st.progress` and `st.status` feedback as it iterates through pages.
   - Master Data Table displayed directly on the UI using `st.dataframe`.
   - Instant one-click download for the **Master BOQ Excel Workbook**.
   - Interactive tabs for Master Summary, Itemized Schedule, Global Material Nesting, and Drawing QC Audit.

---

## 📁 Project Structure

```text
acoustic_takeoff/
├── data/
│   └── sample_acoustic_dwg.pdf       # Multi-page test drawing set (Boardroom, Cover, Auditorium)
├── output/
│   └── Master_BOQ_Verification.xlsx  # Master BOQ Excel spreadsheet
├── modules/
│   ├── __init__.py                   # Package exports & loader
│   ├── 1_pdf_parser.py               # Multi-page batch scanner & color material classifier
│   ├── 2_geometry.py                 # Multi-page geometry takeoff & cutout deduction
│   ├── 3_nesting.py                  # Global 2D bin packing independently by material
│   ├── 4_exporter.py                 # Master BOQ Excel exporter (Summary, Detailed BOQ, Nesting)
│   └── 5_qc_validator.py             # Contractor drafting QC & error detection engine
├── app.py                            # Streamlit web user interface
├── README.md                         # Documentation & user guide
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
```bash
cd acoustic_takeoff
streamlit run app.py
```

### 3. Running Multi-Page Takeoff
1. In the sidebar, click **⚡ 3-Page Set (With QC)** to load the multi-room test project, or upload your own multi-page PDF.
2. Click **🚀 Run Full Project Takeoff** to batch-process all pages with real-time progress.
3. Review the **Master Material Summary Table** and **Page-by-Page Scope Breakdown**.
4. Download the comprehensive **Master BOQ Excel Report (`.xlsx`)**.
