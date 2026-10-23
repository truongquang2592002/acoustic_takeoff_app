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
ai_vision_qs = importlib.import_module(".6_ai_vision_qs", package=__name__)
cad_parser = importlib.import_module(".7_cad_parser", package=__name__)
vision_engine = importlib.import_module(".8_vision_engine", package=__name__)
crewai_qs = importlib.import_module(".9_crewai_qs_team", package=__name__)

# Re-export key dataclasses and functions for clean, top-level usage
VectorPath = pdf_parser.VectorPath
PDFPageData = pdf_parser.PDFPageData
extract_pdf_data = pdf_parser.extract_pdf_data
generate_sample_contractor_drawing = pdf_parser.generate_sample_contractor_drawing
detect_drawing_scales = pdf_parser.detect_drawing_scales
compute_scale_multiplier = pdf_parser.compute_scale_multiplier
is_takeoff_eligible_page = pdf_parser.is_takeoff_eligible_page
is_ap_polyester_material = pdf_parser.is_ap_polyester_material
extract_numeric_dimensions = pdf_parser.extract_numeric_dimensions
calibrate_panel_local_scale_directional = pdf_parser.calibrate_panel_local_scale_directional

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

# AI Vision QS
CutoutItem = ai_vision_qs.CutoutItem
AcousticPanelTakeoffItem = ai_vision_qs.AcousticPanelTakeoffItem
ElevationPageTakeoffResult = ai_vision_qs.ElevationPageTakeoffResult
AIVisionProjectReport = ai_vision_qs.AIVisionProjectReport
render_pdf_page_to_highres_image = ai_vision_qs.render_pdf_page_to_highres_image
extract_cad_ocr_tokens = ai_vision_qs.extract_cad_ocr_tokens
process_document_with_ai_vision_qs = ai_vision_qs.process_document_with_ai_vision_qs
convert_ai_vision_takeoff_to_panel_items = ai_vision_qs.convert_ai_vision_takeoff_to_panel_items

# Native CAD 1:1 Parser
parse_cad_dxf = cad_parser.parse_cad_dxf
generate_sample_cad_dxf = cad_parser.generate_sample_cad_dxf
CADTakeoffSummary = cad_parser.CADTakeoffSummary

# Computer Vision & SAM 2
execute_vipro_vision_pipeline = vision_engine.execute_vipro_vision_pipeline
render_vision_contour_overlay = vision_engine.render_vision_contour_overlay
detect_wall_and_cutouts_yolo = vision_engine.detect_wall_and_cutouts_yolo
predict_sam2_mask_replicate = vision_engine.predict_sam2_mask_replicate
VisionSegmentedPanel = vision_engine.VisionSegmentedPanel

# CrewAI QS Team
ViproCrewQSTeam = crewai_qs.ViproCrewQSTeam
CrewQSTeamOutput = crewai_qs.CrewQSTeamOutput
AgentThoughtLog = crewai_qs.AgentThoughtLog
convert_crewai_output_to_panel_items = crewai_qs.convert_crewai_output_to_panel_items

__all__ = [
    "pdf_parser",
    "geometry",
    "nesting",
    "exporter",
    "qc_validator",
    "ai_vision_qs",
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
    "CutoutItem",
    "AcousticPanelTakeoffItem",
    "ElevationPageTakeoffResult",
    "AIVisionProjectReport",
    "render_pdf_page_to_highres_image",
    "extract_cad_ocr_tokens",
    "process_document_with_ai_vision_qs",
    "convert_ai_vision_takeoff_to_panel_items",
]


