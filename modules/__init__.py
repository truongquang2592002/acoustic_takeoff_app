"""
Acoustic Takeoff & Nesting Engine Modules
"""

import importlib

# Dynamically import numeric-prefixed modules
pdf_parser = importlib.import_module(".1_pdf_parser", package=__name__)
geometry = importlib.import_module(".2_geometry", package=__name__)
nesting = importlib.import_module(".3_nesting", package=__name__)
exporter = importlib.import_module(".4_exporter", package=__name__)
qc_validator = importlib.import_module(".5_qc_validator", package=__name__)

# Re-export key dataclasses and functions for clean, top-level usage
VectorPath = pdf_parser.VectorPath
PDFPageData = pdf_parser.PDFPageData
extract_pdf_data = pdf_parser.extract_pdf_data
generate_sample_contractor_drawing = pdf_parser.generate_sample_contractor_drawing
detect_drawing_scales = pdf_parser.detect_drawing_scales
compute_scale_multiplier = pdf_parser.compute_scale_multiplier

scan_full_document = pdf_parser.scan_full_document

PanelItem = geometry.PanelItem
clean_polygon = geometry.clean_polygon
process_geometry_takeoff = geometry.process_geometry_takeoff
subdivide_large_surface = geometry.subdivide_large_surface
reconstruct_polygons_from_lines = geometry.reconstruct_polygons_from_lines

NestedPiece = nesting.NestedPiece
NestedSheet = nesting.NestedSheet
NestingResult = nesting.NestingResult
GlobalNestingResult = nesting.GlobalNestingResult
OffcutRemnant = nesting.OffcutRemnant
run_nesting_optimization = nesting.run_nesting_optimization
render_nested_sheet_diagram = nesting.render_nested_sheet_diagram

export_takeoff_to_excel = exporter.export_takeoff_to_excel

QCIssue = qc_validator.QCIssue
QCReport = qc_validator.QCReport
run_qc_validation = qc_validator.run_qc_validation
render_qc_overlay_diagram = qc_validator.render_qc_overlay_diagram
render_issue_zoom_crop = qc_validator.render_issue_zoom_crop

__all__ = [
    "pdf_parser",
    "geometry",
    "nesting",
    "exporter",
    "qc_validator",
    "VectorPath",
    "PDFPageData",
    "extract_pdf_data",
    "scan_full_document",
    "generate_sample_contractor_drawing",
    "detect_drawing_scales",
    "compute_scale_multiplier",
    "PanelItem",
    "clean_polygon",
    "process_geometry_takeoff",
    "subdivide_large_surface",
    "reconstruct_polygons_from_lines",
    "NestedPiece",
    "NestedSheet",
    "NestingResult",
    "GlobalNestingResult",
    "OffcutRemnant",
    "run_nesting_optimization",
    "render_nested_sheet_diagram",
    "export_takeoff_to_excel",
    "QCIssue",
    "QCReport",
    "run_qc_validation",
    "render_qc_overlay_diagram",
    "render_issue_zoom_crop",
]


