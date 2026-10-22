"""
Module 1: PDF Vector Parser & Multi-Page Batch Scanner
Extracts vector paths, color layers, text annotations, sheet/room metadata,
and drawing scales across multi-page architectural/contractor PDFs.
"""

from dataclasses import dataclass, field
import io
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import unicodedata

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class VectorPath:
    """Represents an extracted 2D vector path/segment from a PDF drawing."""
    id: int
    command_type: str  # 'rect', 'lines', 'curve', 'polygon'
    points: List[Tuple[float, float]]  # In PDF points (72 pt = 1 inch)
    is_closed: bool
    stroke_color: Optional[Tuple[float, float, float]]  # Normalized (R, G, B) 0-1
    fill_color: Optional[Tuple[float, float, float]]    # Normalized (R, G, B) 0-1
    stroke_width: float
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in pt
    layer: Optional[str] = None
    classified_material: Optional[str] = None
    local_scale: Optional[float] = None
    scale_source: str = "sheet_default"
    dim_w_mm: Optional[float] = None
    dim_h_mm: Optional[float] = None

    @property
    def stroke_hex(self) -> Optional[str]:
        if not self.stroke_color:
            return None
        r, g, b = [int(round(c * 255)) for c in self.stroke_color[:3]]
        return f"#{r:02X}{g:02X}{b:02X}"

    @property
    def fill_hex(self) -> Optional[str]:
        if not self.fill_color:
            return None
        r, g, b = [int(round(c * 255)) for c in self.fill_color[:3]]
        return f"#{r:02X}{g:02X}{b:02X}"


@dataclass
class PDFPageData:
    """Container for vector geometry and metadata extracted from a PDF page."""
    page_number: int
    total_pages: int
    width_pt: float
    height_pt: float
    paths: List[VectorPath] = field(default_factory=list)
    text_blocks: List[Dict[str, Any]] = field(default_factory=list)
    detected_scales: List[Dict[str, Any]] = field(default_factory=list)
    unique_stroke_colors: List[str] = field(default_factory=list)
    unique_fill_colors: List[str] = field(default_factory=list)
    preview_image: Optional[Any] = None  # PIL Image
    sheet_name: str = "Drawing Sheet"
    dynamic_palette: Dict[str, List[str]] = field(default_factory=dict)
    cutout_colors: List[str] = field(default_factory=list)
    legend_bounds: Optional[Tuple[float, float, float, float]] = None
    legend_entries: List[Dict[str, Any]] = field(default_factory=list)
    is_skipped: bool = False
    skip_reason: str = ""


# Standard AEC Acoustic Material Color Palette Mappings
# STRICT QS RULE: Exclusively for Acoustic Polyester (AP / TP605 / TP603 / TP606).
# Excludes Wood Slat (gỗ nan), drywalls, and general non-acoustic finishes.
DEFAULT_MATERIAL_PALETTE: Dict[str, List[str]] = {
    "Polyester 9mm (Acoustic Panel)": [
        "#0078D4", "#0284C7", "#00AAFF", "#0047D4", "#005A9E", "#0077D4", "#38BDF8", "#0369A1",
        "#808080", "#7F7F7F", "#828282", "#A0A0A0", "#9E9E9E", "#6B7280", "#4B5563"  # Acoustic Grey PET Felt
    ],
    "Fabric Wrapped 25mm (Polyester Core)": [
        "#059669", "#10B981", "#047857", "#065F46", "#34D399", "#15803D", "#16A34A"
    ]
}

# Cutout / Void Deduction Colors (Red, Crimson, Scarlet)
DEFAULT_CUTOUT_COLORS: List[str] = [
    "#D92626", "#EF4444", "#DC2626", "#FF0000", "#E11D48", "#B91C1C", "#991B1B", "#F87171", "#DB2626"
]

# AEC Material Recognition Regular Expressions (Bilingual: Vietnamese & English AEC standards)
LEGEND_HEADER_REGEX = re.compile(
    r"(?i)\b(LEGEND|SPECIFICATIONS?|FINISH\s*SCHEDULE|MATERIAL\s*SCHEDULE|"
    r"BANG\s*GHI\s*CHU|GHI\s*CHU|CHU\s*THICH|KY\s*HIEU)\b"
)

MATERIAL_KEYWORDS_REGEX = re.compile(
    r"(?i)\b("
    r"TIEU\s*AM|TAM\s*TIEU\s*AM|VACH\s*TIEU\s*AM|TRAN\s*TIEU\s*AM|"
    r"BOC\s*VAI|VAI\s*NI|GO\s*TIEU\s*AM|SOI\s*KHOANG|MUT\s*HAP\s*THU|MUT\s*TIEU\s*AM|"
    r"DUC\s*LO|XE\s*RANH|PET\s*FELT|POLYESTER|"
    r"ACOUSTIC|WOOD\s*SLAT|FABRIC\s*WRAPPED|SOUND\s*ABSORB|"
    r"BAFFLE|CEILING\s*CLOUD|WALL\s*PANEL|CEILING\s*PANEL|"
    r"TP\s*[-_]?\d{3,4}|AP\s*[-_]?\d*|AC\s*[-_]?\d*|TA\s*[-_]?\d*|WP\s*[-_]?\d*|CP\s*[-_]?\d*|"
    r"BF\s*[-_]?\d*|PN\s*[-_]?\d*"
    r")\b"
)

CUTOUT_KEYWORDS_REGEX = re.compile(
    r"(?i)\b("
    r"CUTOUT|VOID|OPENING|LO\s*KHOET|LO\s*MO|KHOET\s*LO|"
    r"DOOR\s*VOID|WINDOW\s*VOID|GLAZING|CUA\s*DI|CUA\s*SO"
    r")\b"
)


def strip_accents(text: str) -> str:
    """Strip Vietnamese diacritics and convert text to normalized ASCII for robust keyword matching."""
    norm = unicodedata.normalize('NFD', text)
    ascii_text = ''.join(c for c in norm if unicodedata.category(c) != 'Mn')
    return ascii_text


def is_takeoff_eligible_page(sheet_name: str, page_text: str = "") -> Tuple[bool, str]:
    """
    Smart Page Skipping (Chief QS Rule):
    Inspects sheet title and page text:
    - Automatically skips Floor Plans (MẶT BẰNG, FLOOR PLAN, LAYOUT PLAN, MB).
    - Only accepts Elevation / Detail / Section sheets (MẶT ĐỨNG, ELEVATION, CHI TIẾT, DETAIL, SECTION).
    """
    norm_sheet = strip_accents(sheet_name).upper()
    norm_text = strip_accents(page_text[:2500]).upper() if page_text else ""

    fuzzy_plan_regex = re.compile(r"(?i)\b(M[\.·\?]?T\s*B[\.·\?]?NG|FLOOR\s*PLAN|LAYOUT\s*PLAN|FURNITURE\s*PLAN|MB\s*TUONG|MB\s*BO\s*TRI|MB\s*NOI\s*THAT|REFLECTED\s*CEILING|KEY\s*PLAN)\b")
    fuzzy_elevation_regex = re.compile(r"(?i)\b(M[\.·\?]?T\s*[\.·\?]*[DÐ][\.·\?]*NG|CHI\s*TI[\.·\?]*T|ELEVATION|ELEV|DETAIL|SECTION|M[\.·\?]?T\s*C[\.·\?]?T|WALL\s*ELEVATION|FITOUT\s*ELEVATION)\b")

    # Rule 1: Explicit Floor / Layout Plan exclusion
    plan_keywords = [
        "MAT BANG", "FLOOR PLAN", "LAYOUT PLAN", "FURNITURE PLAN",
        "MB TUONG", "MB BO TRI", "MB NOI THAT", "MB TRAN",
        "REFLECTED CEILING PLAN", "KEY PLAN"
    ]
    if any(kw in norm_sheet for kw in plan_keywords) or fuzzy_plan_regex.search(norm_sheet):
        return False, "Bản vẽ Mặt Bằng (Floor/Layout Plan) - Bỏ qua, không bóc diện tích vách"

    # Rule 2: Explicit Elevation / Detail inclusion
    elevation_keywords = [
        "MAT DUNG", "ELEVATION", "CHI TIET", "DETAIL", "SECTION",
        "MAT CAT", "WALL ELEVATION", "FITOUT ELEVATION"
    ]
    if any(kw in norm_sheet for kw in elevation_keywords) or fuzzy_elevation_regex.search(norm_sheet):
        return True, "Bản vẽ Mặt Đứng / Chi Tiết (Elevation/Detail)"

    # Rule 3: Check page body text if sheet title was generic
    if any(kw in norm_text for kw in plan_keywords) or fuzzy_plan_regex.search(norm_text):
        if not (any(ek in norm_text for ek in elevation_keywords) or fuzzy_elevation_regex.search(norm_text)):
            return False, "Nội dung trang chứa từ khóa Mặt Bằng - Bỏ qua"

    if any(kw in norm_text for kw in elevation_keywords) or fuzzy_elevation_regex.search(norm_text):
        return True, "Nội dung trang chứa từ khóa Mặt Đứng / Chi Tiết"

    if "WALL" in norm_text or "VACH" in norm_text:
        return True, "Phát hiện chỉ định diện vách tường (Wall Fitout)"

    # Default fallback for unannotated pages
    return False, "Không phát hiện ký hiệu Mặt Đứng / Chi Tiết - Bỏ qua"


def is_ap_polyester_material(material_name: Optional[str]) -> bool:
    """
    Strict Material Filter (Chief QS Rule):
    Exclusively retains Acoustic Polyester Panels (AP, Tiêu Âm TP605, TP603, TP606, Polyester).
    Rejects Wood Slat (gỗ nan), Timber, Drywall, and non-AP materials.
    """
    if not material_name:
        return False
    norm = strip_accents(material_name).upper()
    pattern = r"(?i)(\bAP\b|\bAP[-\s]?\d*|\bTP605\b|\bTP603\b|\bTP606\b|POLYESTER|TIEU\s*AM)"
    return bool(re.search(pattern, norm))


def extract_numeric_dimensions(page: Any) -> List[Dict[str, Any]]:
    """
    Extract CAD dimension text callouts (e.g. '2400', '1,200', '600') with coordinate bounding boxes.
    Filters out drawing years, sheet numbers, or structural grid designations.
    """
    dims: List[Dict[str, Any]] = []
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return dims

    for b in blocks:
        if len(b) < 5 or not b[4].strip():
            continue
        txt = b[4].strip()
        matches = re.finditer(r"\b(\d{1,2}[,\.]\d{3}|\d{3,4})(?:\s*mm)?\b", txt, re.IGNORECASE)
        for m in matches:
            val_str = m.group(1).replace(",", "").replace(".", "")
            try:
                val_mm = float(val_str)
                # Filter out years, sheet numbers, stc ratings
                if 250.0 <= val_mm <= 15000.0 and val_mm not in (2024.0, 2025.0, 2026.0, 2027.0):
                    dims.append({
                        "val_mm": val_mm,
                        "raw_text": txt,
                        "bbox": (b[0], b[1], b[2], b[3]),
                        "cx": (b[0] + b[2]) / 2.0,
                        "cy": (b[1] + b[3]) / 2.0
                    })
            except ValueError:
                continue
    return dims


def calibrate_panel_local_scale_directional(
    panel_bbox: Tuple[float, float, float, float],
    dim_candidates: List[Dict[str, Any]],
    default_scale_multiplier: float
) -> Tuple[float, str, Optional[float], Optional[float]]:
    """
    Dimension-Driven Local Calibration (Chief QS Algorithm):
    Directionally correlates dimension callouts with panel edges:
    - Horizontal dimension line (spans along X, placed just above/below panel) -> measures Width (pw).
    - Vertical dimension line (spans along Y, placed just left/right of panel) -> measures Height (ph).
    - Checks candidate scale against standard AEC scales (1:10, 1:20, 1:25, 1:50, 1:100).
    Returns: (local_scale_factor, source_label, matched_w_mm, matched_h_mm)
    """
    px0, py0, px1, py1 = panel_bbox
    pw = abs(px1 - px0)
    ph = abs(py1 - py0)

    if pw < 2.0 or ph < 2.0 or not dim_candidates:
        return default_scale_multiplier, "sheet_default", None, None

    std_scales = [1.7639, 3.5278, 7.0556, 8.8194, 17.6389, 35.2778]

    calibrated_scales: List[float] = []
    matched_w_mm: Optional[float] = None
    matched_h_mm: Optional[float] = None

    for d in dim_candidates:
        dcx, dcy = d["cx"], d["cy"]
        val = d["val_mm"]

        # Horizontal Dimension: centered within panel X span, offset vertically just above or below
        is_horiz = (px0 - 10.0 <= dcx <= px1 + 10.0) and (abs(dcy - py0) <= 45.0 or abs(dcy - py1) <= 45.0) and not (py0 + 5.0 <= dcy <= py1 - 5.0)
        if is_horiz:
            cand_scale = val / pw
            for std_s in std_scales:
                if abs(cand_scale - std_s) / std_s <= 0.08:
                    calibrated_scales.append(cand_scale)
                    matched_w_mm = val
                    break

        # Vertical Dimension: centered within panel Y span, offset horizontally just left or right
        is_vert = (py0 - 10.0 <= dcy <= py1 + 10.0) and (abs(dcx - px0) <= 45.0 or abs(dcx - px1) <= 45.0) and not (px0 + 5.0 <= dcx <= px1 - 5.0)
        if is_vert:
            cand_scale = val / ph
            for std_s in std_scales:
                if abs(cand_scale - std_s) / std_s <= 0.08:
                    calibrated_scales.append(cand_scale)
                    matched_h_mm = val
                    break

    if calibrated_scales:
        final_scale = sum(calibrated_scales) / len(calibrated_scales)
        return final_scale, "dim_calibrated", matched_w_mm, matched_h_mm

    return default_scale_multiplier, "sheet_default", None, None


def rgb_to_hex(rgb: Optional[Tuple[float, ...]]) -> Optional[str]:
    """Convert normalized (0-1) RGB tuple to hex string."""
    if not rgb:
        return None
    r, g, b = [int(round(max(0.0, min(1.0, c)) * 255)) for c in rgb[:3]]
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(hex_str: str) -> Tuple[float, float, float]:
    """Convert hex string (e.g. '#00AAFF') to normalized (0-1) RGB tuple."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join([c * 2 for c in hex_str])
    if len(hex_str) != 6:
        return (0.0, 0.0, 0.0)
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b)


def color_distance(hex1: Optional[str], hex2: Optional[str]) -> float:
    """Compute normalized Euclidean RGB distance (0.0 to 1.0) between two hex strings."""
    if not hex1 or not hex2:
        return 1.0
    h1 = hex1.lstrip("#")
    h2 = hex2.lstrip("#")
    if len(h1) != 6 or len(h2) != 6:
        return 1.0
    r1, g1, b1 = int(h1[0:2], 16), int(h1[2:4], 16), int(h1[4:6], 16)
    r2, g2, b2 = int(h2[0:2], 16), int(h2[2:4], 16), int(h2[4:6], 16)
    dist = math.hypot(r1 - r2, g1 - g2, b1 - b2)
    return dist / 441.67295593


def matches_color_list(candidate_hex: Optional[str], target_hexes: List[str], tolerance: float = 0.10) -> bool:
    """Check if candidate hex matches any color in target list within RGB tolerance."""
    if not candidate_hex or not target_hexes:
        return False
    for target in target_hexes:
        if color_distance(candidate_hex, target) <= tolerance:
            return True
    return False


def clean_legend_label(text: str) -> str:
    """Clean and standardize material annotation / legend item text."""
    cleaned = re.sub(r"^(?:TYPE|MÃ|KÝ HIỆU|LOẠI)[:\s\-]+", "", text, flags=re.I).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def extract_page_legend_and_swatches(
    page: Any,
    drawings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Senior QS Legend & Schedule Parser:
    1. Reads text annotations to locate Legend / Ghi Chú headers and determine the legend bounds.
    2. Identifies acoustic material descriptions (e.g. TP605, GỖ TIÊU ÂM, BỌC VẢI, AP-1).
    3. Finds adjacent vector swatches (compact filled rects or stroke indicator lines) on the same horizontal band (dy <= 9 pt).
    4. Dynamically builds a color-to-material mapping dictionary (including grey, brown, or custom CAD colors).
    5. Detects cutout deduction styles (e.g. red lines, dashed openings).
    """
    rect = page.rect
    width_pt, height_pt = rect.width, rect.height
    words = page.get_text("words")

    # 1. Locate Legend Header Block (Accent-insensitive)
    legend_header_pt = None
    legend_header_bbox = None
    blocks = page.get_text("blocks")
    for b in blocks:
        raw_t = b[4].strip()
        norm_t = strip_accents(raw_t).upper()
        if LEGEND_HEADER_REGEX.search(norm_t) or LEGEND_HEADER_REGEX.search(raw_t.upper()):
            legend_header_bbox = (b[0], b[1], b[2], b[3])
            legend_header_pt = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            break

    legend_bounds: Optional[Tuple[float, float, float, float]] = None
    if legend_header_pt:
        enclosing_rects = []
        for d in drawings:
            r = d.get("rect")
            if not r:
                continue
            if r.width > width_pt * 0.85 or r.height > height_pt * 0.85:
                continue
            if r.x0 <= legend_header_pt[0] <= r.x1 and r.y0 <= legend_header_pt[1] <= r.y1:
                enclosing_rects.append(r)
        if enclosing_rects:
            enclosing_rects.sort(key=lambda r: r.width * r.height)
            best_r = enclosing_rects[0]
            legend_bounds = (best_r.x0, best_r.y0, best_r.x1, best_r.y1)
        elif legend_header_bbox:
            legend_bounds = (
                max(0.0, legend_header_bbox[0] - 25.0),
                max(0.0, legend_header_bbox[1] - 15.0),
                min(width_pt, legend_header_bbox[0] + 360.0),
                min(height_pt, legend_header_bbox[3] + 240.0)
            )

    # If this page does NOT contain a legend table, return empty to prevent elevation text from leaking into swatches
    if not legend_bounds:
        return {
            "legend_bounds": None,
            "dynamic_palette": {},
            "cutout_colors": [],
            "legend_entries": []
        }

    lx0, ly0, lx1, ly1 = legend_bounds

    # 2. Group words into lines inside legend bounds
    lines_dict: Dict[Tuple[int, int], List[Any]] = {}
    for w in words:
        if lx0 - 15 <= w[0] and w[2] <= lx1 + 15 and ly0 - 15 <= w[1] and w[3] <= ly1 + 15:
            key = (w[5], w[6])
            lines_dict.setdefault(key, []).append(w)

    legend_text_lines = []
    for (b_no, l_no), w_list in lines_dict.items():
        w_list.sort(key=lambda item: item[7])
        t = " ".join(item[4] for item in w_list).strip()
        if not t:
            continue
        min_x = min(item[0] for item in w_list)
        min_y = min(item[1] for item in w_list)
        max_x = max(item[2] for item in w_list)
        max_y = max(item[3] for item in w_list)

        t_norm = strip_accents(t).upper()
        # Skip legend header line
        if LEGEND_HEADER_REGEX.search(t_norm) and any(w in t_norm for w in ["ACOUSTIC", "SPECIFICATION", "GHI CHU", "CHÚ THÍCH", "LEGEND"]):
            continue

        legend_text_lines.append({
            "text": t,
            "norm_text": t_norm,
            "bbox": (min_x, min_y, max_x, max_y),
            "center_y": (min_y + max_y) / 2.0,
            "center_x": (min_x + max_x) / 2.0
        })

    # 3. Collect candidate swatches (compact shapes) inside legend bounds
    swatches = []
    for d in drawings:
        r = d.get("rect")
        if not r:
            continue
        if not (lx0 - 8 <= r.x0 and r.x1 <= lx1 + 8 and ly0 - 8 <= r.y0 and r.y1 <= ly1 + 8):
            continue
        # Skip legend container border
        if abs(r.width - (lx1 - lx0)) < 6.0 and abs(r.height - (ly1 - ly0)) < 6.0:
            continue

        w, h = r.width, r.height
        if (4.0 <= w <= 140.0 and 0.0 <= h <= 50.0) or (0.0 <= w <= 50.0 and 4.0 <= h <= 140.0):
            stroke_h = rgb_to_hex(d.get("color"))
            fill_h = rgb_to_hex(d.get("fill"))
            if fill_h == "#FFFFFF" and stroke_h in (None, "#000000", "#FFFFFF"):
                continue
            swatches.append({
                "rect": r,
                "center_y": (r.y0 + r.y1) / 2.0,
                "stroke_hex": stroke_h,
                "fill_hex": fill_h,
                "d_ref": d
            })

    dynamic_palette: Dict[str, List[str]] = {}
    cutout_colors: List[str] = []
    legend_entries: List[Dict[str, Any]] = []

    # 4. Proximity alignment between text and swatches
    for line in legend_text_lines:
        txt = line["text"]
        t_norm = line["norm_text"]
        is_cutout = bool(CUTOUT_KEYWORDS_REGEX.search(t_norm)) or bool(CUTOUT_KEYWORDS_REGEX.search(txt))
        is_mat = bool(MATERIAL_KEYWORDS_REGEX.search(t_norm)) or bool(MATERIAL_KEYWORDS_REGEX.search(txt))
        if not (is_cutout or is_mat):
            continue

        l_x0, l_y0, l_x1, l_y1 = line["bbox"]
        l_cy = line["center_y"]

        best_swatch = None
        best_dist = 9999.0
        for s in swatches:
            dy = abs(s["center_y"] - l_cy)
            if dy > 9.0:
                continue
            sr = s["rect"]
            if sr.x1 <= l_x0 + 15 and (l_x0 - sr.x1) <= 140:
                dx = max(0.0, l_x0 - sr.x1)
            elif sr.x0 >= l_x1 - 15 and (sr.x0 - l_x1) <= 140:
                dx = max(0.0, sr.x0 - l_x1)
            else:
                continue
            score = dx + dy * 3.0
            if score < best_dist:
                best_dist = score
                best_swatch = s

        if best_swatch:
            s_fill = best_swatch["fill_hex"]
            s_stroke = best_swatch["stroke_hex"]
            chosen_hexes = []
            if s_fill and s_fill != "#FFFFFF":
                chosen_hexes.append(s_fill)
            if s_stroke and s_stroke not in chosen_hexes and s_stroke != "#FFFFFF":
                chosen_hexes.append(s_stroke)

            if is_cutout:
                for c in chosen_hexes:
                    if c not in cutout_colors:
                        cutout_colors.append(c)
                legend_entries.append({
                    "type": "cutout",
                    "label": txt,
                    "hex_colors": chosen_hexes
                })
            elif is_mat:
                # QS Strict Filter: ONLY register material if text satisfies AP/Polyester/Tiêu Âm criteria
                if not is_ap_polyester_material(txt):
                    continue
                clean_name = clean_legend_label(txt)
                if chosen_hexes:
                    dynamic_palette.setdefault(clean_name, [])
                    for c in chosen_hexes:
                        if c not in dynamic_palette[clean_name]:
                            dynamic_palette[clean_name].append(c)
                    legend_entries.append({
                        "type": "material",
                        "label": clean_name,
                        "hex_colors": chosen_hexes
                    })

    return {
        "legend_bounds": legend_bounds,
        "dynamic_palette": dynamic_palette,
        "cutout_colors": cutout_colors,
        "legend_entries": legend_entries
    }


def classify_vector_material(
    stroke_hex: Optional[str],
    fill_hex: Optional[str],
    palette_map: Optional[Dict[str, List[str]]] = None,
    cutout_colors: Optional[List[str]] = None,
    tolerance: float = 0.10
) -> Tuple[Optional[str], bool]:
    """
    Classify acoustic material type based on line color, hatching, or fill using
    dynamic palette matching, CAD color tolerance (Euclidean distance), and void detection.
    Strictly restricted to AP Polyester panels (TP605, TP603, TP606, Polyester).
    Returns: (material_type or None, is_cutout: bool)
    """
    palette = palette_map if palette_map is not None else DEFAULT_MATERIAL_PALETTE
    active_cutouts = (cutout_colors or []) + DEFAULT_CUTOUT_COLORS

    # 1. Check if it's a cutout / opening deduction
    for hex_val in (stroke_hex, fill_hex):
        if hex_val and matches_color_list(hex_val, active_cutouts, tolerance=tolerance):
            return ("Cutout / Opening", True)

    test_hex = fill_hex or stroke_hex
    if not test_hex:
        return (None, False)

    # 2. Check explicitly mapped materials in palette (fill prioritized over stroke)
    if fill_hex:
        for mat_name, hex_list in palette.items():
            if matches_color_list(fill_hex, hex_list, tolerance=tolerance):
                if is_ap_polyester_material(mat_name):
                    return (mat_name, False)

    if stroke_hex:
        for mat_name, hex_list in palette.items():
            if matches_color_list(stroke_hex, hex_list, tolerance=tolerance):
                if is_ap_polyester_material(mat_name):
                    return (mat_name, False)

    # 3. Filter out unmapped structural walls, grid lines, and title block borders
    r, g, b = hex_to_rgb(test_hex)
    max_c, min_c = max(r, g, b), min(r, g, b)
    is_grayscale = (max_c - min_c < 0.12) or (max_c < 0.28)
    if is_grayscale:
        # Ignore unmapped structural walls, grid lines, and title block borders
        return (None, False)

    # 4. Heuristic Hue Fallback for unmapped colored lines (Polyester only)
    if b > r + 0.15 and b > g + 0.10:
        return ("Polyester 9mm (Acoustic Panel)", False)
    elif g > r + 0.15 and g > b + 0.12:
        return ("Fabric Wrapped 25mm (Polyester Core)", False)

    return (None, False)



def detect_drawing_scales(text_content: str) -> List[Dict[str, Any]]:
    """
    Search extracted PDF text for architectural and engineering scale annotations.
    Examples: 1:50, 1:100, 1/4" = 1'-0", 1/8" = 1'-0", 3/16" = 1'-0"
    """
    scales = []

    # 1. Metric ratio scales (e.g. 1:50, 1:100, 1:20, 1:200, 1:25)
    metric_matches = re.finditer(r"(?:SCALE[:\s]*|NTS[:\s]*)?1\s*:\s*(\d{1,4})\b", text_content, re.IGNORECASE)
    for m in metric_matches:
        ratio = int(m.group(1))
        mm_per_pt = ratio * (25.4 / 72.0)
        scales.append({
            "type": "metric",
            "label": f"1:{ratio}",
            "ratio": ratio,
            "mm_per_pt": mm_per_pt,
            "raw_text": m.group(0).strip()
        })

    # 2. Imperial architectural scales (e.g. 1/4" = 1'-0", 1/8" = 1'-0", 3/8" = 1'-0")
    imperial_patterns = [
        (r'(\d+/\d+)\s*["\u201d]?\s*=\s*1\s*[\'’]\s*-\s*0\s*["\u201d]?', "arch"),
        (r'(\d+/\d+)\s*["\u201d]?\s*=\s*1\s*[\'’]', "arch"),
        (r'(\d+)\s*["\u201d]?\s*=\s*1\s*[\'’]\s*-\s*0\s*["\u201d]?', "arch_int")
    ]
    for pat, _ in imperial_patterns:
        for m in re.finditer(pat, text_content, re.IGNORECASE):
            frac_str = m.group(1)
            try:
                if "/" in frac_str:
                    num, den = frac_str.split("/")
                    inches_on_paper = float(num) / float(den)
                else:
                    inches_on_paper = float(frac_str)
                mm_per_pt = 304.8 / (inches_on_paper * 72.0)
                scales.append({
                    "type": "imperial",
                    "label": f'{frac_str}" = 1\'-0"',
                    "ratio": round(12.0 / inches_on_paper, 2),
                    "mm_per_pt": mm_per_pt,
                    "raw_text": m.group(0).strip()
                })
            except Exception:
                continue

    # Deduplicate by label
    unique_scales = []
    seen = set()
    for s in scales:
        if s["label"] not in seen:
            seen.add(s["label"])
            unique_scales.append(s)

    return unique_scales


def compute_scale_multiplier(
    scale_preset: str,
    target_unit: str = "mm",
    custom_drawing_pts: float = 72.0,
    custom_real_units: float = 1000.0
) -> float:
    """
    Returns the multiplier to convert PDF coordinates (points) into the target real-world unit.
    E.g. If result is 17.639, then (coord_pt * 17.639) = length in mm.
    """
    pt_in_mm_base = 25.4 / 72.0

    preset_mm_factors = {
        "1:1 (Full Scale)": pt_in_mm_base * 1,
        "1:10": pt_in_mm_base * 10,
        "1:20": pt_in_mm_base * 20,
        "1:25": pt_in_mm_base * 25,
        "1:50": pt_in_mm_base * 50,
        "1:100": pt_in_mm_base * 100,
        "1:200": pt_in_mm_base * 200,
        "1:500": pt_in_mm_base * 500,
        '1/8" = 1\'-0" (1:96)': pt_in_mm_base * 96,
        '3/16" = 1\'-0" (1:64)': pt_in_mm_base * 64,
        '1/4" = 1\'-0" (1:48)': pt_in_mm_base * 48,
        '3/8" = 1\'-0" (1:32)': pt_in_mm_base * 32,
        '1/2" = 1\'-0" (1:24)': pt_in_mm_base * 24,
        '3/4" = 1\'-0" (1:16)': pt_in_mm_base * 16,
        '1" = 1\'-0" (1:12)': pt_in_mm_base * 12,
        '1-1/2" = 1\'-0" (1:8)': pt_in_mm_base * 8,
        '3" = 1\'-0" (1:4)': pt_in_mm_base * 4,
    }

    if scale_preset == "Custom Calibration":
        if custom_drawing_pts <= 0:
            return 1.0
        return custom_real_units / custom_drawing_pts

    mm_per_pt = preset_mm_factors.get(scale_preset, pt_in_mm_base * 50)

    if target_unit == "mm":
        return mm_per_pt
    elif target_unit == "m":
        return mm_per_pt / 1000.0
    elif target_unit == "inch":
        return mm_per_pt / 25.4
    elif target_unit == "ft":
        return mm_per_pt / 304.8
    return mm_per_pt


def extract_sheet_or_room_name(page: Any) -> str:
    """
    Use page.get_text("words") to auto-detect Sheet Name or Room Name
    by examining word dimensions, line grouping, title block areas, and room keyword patterns.
    """
    try:
        words = page.get_text("words")
    except Exception:
        words = []

    if not words:
        return f"Sheet {page.number + 1}"

    # Structure of each word: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    rect = page.rect
    width_pt, height_pt = rect.width, rect.height

    # 1. Group words by (block_no, line_no) to assemble text lines with bounding box and height
    lines_dict: Dict[Tuple[int, int], List[Any]] = {}
    for w in words:
        key = (w[5], w[6])
        lines_dict.setdefault(key, []).append(w)

    assembled_lines: List[Dict[str, Any]] = []
    for (b_no, l_no), w_list in lines_dict.items():
        w_list.sort(key=lambda item: item[7])  # sort by word_no
        text = " ".join(item[4] for item in w_list).strip()
        if not text:
            continue
        min_x = min(item[0] for item in w_list)
        min_y = min(item[1] for item in w_list)
        max_x = max(item[2] for item in w_list)
        max_y = max(item[3] for item in w_list)
        avg_h = sum(item[3] - item[1] for item in w_list) / max(1, len(w_list))
        assembled_lines.append({
            "text": text,
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "height": avg_h,
            "in_title_block": (min_x > width_pt * 0.45 and min_y > height_pt * 0.65)
        })

    # Priority 1: Check for explicit "DRAWING:" or "TITLE:" in title block
    for line in assembled_lines:
        t_upper = line["text"].upper()
        if "DRAWING:" in t_upper or "DWG TITLE:" in t_upper or "SHEET TITLE:" in t_upper:
            cleaned = re.sub(r"^(?:DRAWING|DWG TITLE|SHEET TITLE)[:\s\-]+", "", line["text"], flags=re.I).strip()
            if len(cleaned) >= 3:
                return cleaned

    # Priority 2: Check for explicit Room / Studio / Hall names in large font
    room_keywords = ["BOARDROOM", "STUDIO", "AUDITORIUM", "CONFERENCE", "OFFICE", "BALLROOM", "THEATER", "HALL", "SUITE", "ATRIUM", "FITOUT"]
    candidates = []
    for line in assembled_lines:
        t_upper = line["text"].upper()
        for kw in room_keywords:
            if kw in t_upper and len(line["text"]) < 80:
                candidates.append((line["height"], line["text"]))
                break

    if candidates:
        # Pick the one with largest font height
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]

    # Priority 3: Large text in title block
    title_block_lines = [l for l in assembled_lines if l["in_title_block"] and l["height"] >= 8.0]
    if title_block_lines:
        title_block_lines.sort(key=lambda l: l["height"], reverse=True)
        for l in title_block_lines:
            txt = l["text"].strip()
            if len(txt) > 5 and not txt.startswith("DATE") and not txt.startswith("SCALE"):
                return txt

    # Priority 4: Any prominent headline line (top 15% height or largest height across page)
    assembled_lines.sort(key=lambda l: l["height"], reverse=True)
    for l in assembled_lines:
        if len(l["text"]) >= 4 and l["height"] >= 10.0:
            return l["text"]

    return f"Sheet {page.number + 1}"


def extract_pdf_data(
    file_source: Union[str, Path, bytes, io.BytesIO],
    page_number: int = 0,
    render_dpi: int = 150
) -> PDFPageData:
    """
    Parse vector drawings, text annotations, and render a high-res raster preview from a PDF page.
    Automatically reads Legend / Schedule to construct dynamic material palette and deduce cutouts.
    """
    if fitz is None:
        raise ImportError("PyMuPDF (`fitz`) is required. Please install it with `pip install pymupdf`.")

    if isinstance(file_source, (str, Path)):
        doc = fitz.open(str(file_source))
    elif isinstance(file_source, bytes):
        doc = fitz.open(stream=file_source, filetype="pdf")
    elif isinstance(file_source, io.BytesIO):
        doc = fitz.open(stream=file_source.getvalue(), filetype="pdf")
    else:
        raise TypeError(f"Unsupported file source type: {type(file_source)}")

    total_pages = len(doc)
    if total_pages == 0:
        raise ValueError("PDF document has 0 pages.")

    page_idx = max(0, min(page_number, total_pages - 1))
    page = doc[page_idx]
    rect = page.rect
    width_pt, height_pt = rect.width, rect.height

    preview_img = None
    try:
        pix = page.get_pixmap(dpi=render_dpi)
        preview_img = Image.open(io.BytesIO(pix.tobytes("png")))
    except Exception as e:
        print(f"Warning: Failed to render preview image: {e}")

    drawings = page.get_drawings()
    extracted_paths: List[VectorPath] = []
    stroke_colors_set = set()
    fill_colors_set = set()

    # 1. Page Eligibility & Metadata
    raw_text = page.get_text("text")
    sheet_name = extract_sheet_or_room_name(page)
    is_eligible, skip_reason = is_takeoff_eligible_page(sheet_name, raw_text)
    dim_candidates = extract_numeric_dimensions(page) if is_eligible else []

    # 2. Senior QS Legend & Swatch Analysis
    legend_info = extract_page_legend_and_swatches(page, drawings)
    page_palette = legend_info["dynamic_palette"]
    page_cutouts = legend_info["cutout_colors"]
    legend_bounds = legend_info["legend_bounds"]

    # Merge dynamic page palette on top of default palette
    active_palette: Dict[str, List[str]] = dict(DEFAULT_MATERIAL_PALETTE)
    for mat_name, hex_list in page_palette.items():
        active_palette[mat_name] = hex_list

    path_id = 1
    # Only process drawing paths if page is eligible (not a floor plan)
    if is_eligible:
        for d in drawings:
            stroke = d.get("color")
            fill = d.get("fill")
            width = d.get("width", 1.0)
            layer = d.get("layer")
            d_rect = d.get("rect", (0, 0, 0, 0))

            # Skip swatch icons and border lines inside the Legend container so they aren't taken off as room panels
            if legend_bounds:
                lx0, ly0, lx1, ly1 = legend_bounds
                if lx0 - 4 <= d_rect[0] and d_rect[2] <= lx1 + 4 and ly0 - 4 <= d_rect[1] and d_rect[3] <= ly1 + 4:
                    continue

            stroke_h = rgb_to_hex(stroke)
            if stroke_h:
                stroke_colors_set.add(stroke_h)

            fill_h = rgb_to_hex(fill)
            if fill_h:
                fill_colors_set.add(fill_h)

            items = d.get("items", [])
            pts = []
            is_closed = d.get("closePath", False)
            cmd_type = "lines"

            for it in items:
                cmd = it[0]
                if cmd == "l":
                    p1, p2 = it[1], it[2]
                    if not pts:
                        pts.append((p1.x, p1.y))
                    pts.append((p2.x, p2.y))
                elif cmd == "re":
                    r = it[1]
                    pts = [
                        (r.x0, r.y0),
                        (r.x1, r.y0),
                        (r.x1, r.y1),
                        (r.x0, r.y1),
                        (r.x0, r.y0)
                    ]
                    is_closed = True
                    cmd_type = "rect"
                elif cmd == "c":
                    p1, p2, p3, p4 = it[1], it[2], it[3], it[4]
                    if not pts:
                        pts.append((p1.x, p1.y))
                    for t_step in [0.33, 0.66, 1.0]:
                        inv = 1.0 - t_step
                        bx = (inv**3)*p1.x + 3*(inv**2)*t_step*p2.x + 3*inv*(t_step**2)*p3.x + (t_step**3)*p4.x
                        by = (inv**3)*p1.y + 3*(inv**2)*t_step*p2.y + 3*inv*(t_step**2)*p3.y + (t_step**3)*p4.y
                        pts.append((bx, by))
                    cmd_type = "curve"

            if pts:
                if len(pts) >= 3 and math.isclose(pts[0][0], pts[-1][0], abs_tol=1e-3) and math.isclose(pts[0][1], pts[-1][1], abs_tol=1e-3):
                    is_closed = True

                bbox_tuple = (d_rect[0], d_rect[1], d_rect[2], d_rect[3]) if hasattr(d_rect, "__getitem__") else (0, 0, 0, 0)

                # Classify material using dynamic palette, CAD tolerance, and cutout styles
                mat_type, _ = classify_vector_material(
                    stroke_h, fill_h,
                    palette_map=active_palette,
                    cutout_colors=page_cutouts
                )

                # Local Scale Calibration via nearby Dimension Text
                loc_scale, sc_src, d_w, d_h = calibrate_panel_local_scale_directional(
                    bbox_tuple, dim_candidates, compute_scale_multiplier("1:50")
                )

                extracted_paths.append(
                    VectorPath(
                        id=path_id,
                        command_type=cmd_type,
                        points=pts,
                        is_closed=is_closed,
                        stroke_color=stroke[:3] if stroke else None,
                        fill_color=fill[:3] if fill else None,
                        stroke_width=width,
                        bbox=bbox_tuple,
                        layer=layer,
                        classified_material=mat_type,
                        local_scale=loc_scale,
                        scale_source=sc_src,
                        dim_w_mm=d_w,
                        dim_h_mm=d_h
                    )
                )
                path_id += 1

    blocks = page.get_text("blocks")
    text_data = []
    for b in blocks:
        if len(b) >= 5 and b[4].strip():
            text_data.append({
                "bbox": (b[0], b[1], b[2], b[3]),
                "text": b[4].strip()
            })

    detected_scales = detect_drawing_scales(raw_text)

    doc.close()

    return PDFPageData(
        page_number=page_idx,
        total_pages=total_pages,
        width_pt=width_pt,
        height_pt=height_pt,
        paths=extracted_paths,
        text_blocks=text_data,
        detected_scales=detected_scales,
        unique_stroke_colors=sorted(list(stroke_colors_set)),
        unique_fill_colors=sorted(list(fill_colors_set)),
        preview_image=preview_img,
        sheet_name=sheet_name,
        dynamic_palette=page_palette,
        cutout_colors=page_cutouts,
        legend_bounds=legend_bounds,
        legend_entries=legend_info.get("legend_entries", []),
        is_skipped=not is_eligible,
        skip_reason=skip_reason
    )


def scan_full_document(
    pdf_path: Union[str, Path, bytes, io.BytesIO],
    scale_factor: float = 17.6389,
    material_color_map: Optional[Dict[str, List[str]]] = None,
    render_previews: bool = False,
    preview_dpi: int = 100
) -> Dict[str, Any]:
    """
    Multi-Page Batch Scanner (Senior QS Workflow):
    1. Reads Bảng Ghi Chú (Legend) across all pages to extract dynamic material mappings (including grey, brown).
    2. Maintains a document-wide cumulative palette so Master Legends on Sheet 1 propagate to all elevations.
    3. Auto-detects Sheet Name or Room Name per page using `page.get_text("words")`.
    4. Quét toàn bộ mặt đứng: matches vectors against dynamic palette within CAD RGB Euclidean tolerance (<= 0.10).
    5. Deducts cutouts / opening deduction vectors page-by-page.
    6. Returns a structured nested dictionary grouped by Page_Number and Material_Type compatible with geometry engine.
    """
    if fitz is None:
        raise ImportError("PyMuPDF (`fitz`) is required. Please install it with `pip install pymupdf`.")

    if isinstance(pdf_path, (str, Path)):
        doc = fitz.open(str(pdf_path))
        doc_name = Path(pdf_path).name
    elif isinstance(pdf_path, bytes):
        doc = fitz.open(stream=pdf_path, filetype="pdf")
        doc_name = "Uploaded_Project_Drawing.pdf"
    elif isinstance(pdf_path, io.BytesIO):
        doc = fitz.open(stream=pdf_path.getvalue(), filetype="pdf")
        doc_name = "Stream_Project_Drawing.pdf"
    else:
        raise TypeError(f"Unsupported file source type: {type(pdf_path)}")

    total_pages = len(doc)

    # Document-wide cumulative palette: Master Legend on Sheet 1 applies to the whole package
    global_palette: Dict[str, List[str]] = dict(DEFAULT_MATERIAL_PALETTE)
    if material_color_map is not None:
        for m_name, hexes in material_color_map.items():
            global_palette[m_name] = hexes

    global_cutouts: List[str] = list(DEFAULT_CUTOUT_COLORS)

    pages_result: Dict[int, Dict[str, Any]] = {}
    all_materials_encountered = set()
    total_acoustic_paths_found = 0

    for page_idx in range(total_pages):
        page = doc[page_idx]
        rect = page.rect
        width_pt, height_pt = rect.width, rect.height
        page_num = page_idx + 1

        # 1. Sheet Name & Page Eligibility Analysis (Smart Page Skipping)
        sheet_name = extract_sheet_or_room_name(page)
        raw_text = page.get_text("text")
        detected_scales = detect_drawing_scales(raw_text)

        is_eligible, skip_reason = is_takeoff_eligible_page(sheet_name, raw_text)
        if not is_eligible:
            # Gracefully skip floor plan / layout plan
            preview_img = None
            if render_previews:
                try:
                    pix = page.get_pixmap(dpi=preview_dpi)
                    preview_img = Image.open(io.BytesIO(pix.tobytes("png")))
                except Exception:
                    preview_img = None

            pages_result[page_num] = {
                "page_number": page_num,
                "page_index": page_idx,
                "sheet_name": sheet_name,
                "detected_scales": detected_scales,
                "width_pt": width_pt,
                "height_pt": height_pt,
                "materials": {},
                "cutout_paths": [],
                "all_paths": [],
                "total_material_paths": 0,
                "preview_image": preview_img,
                "dynamic_palette": {},
                "cutout_colors": [],
                "legend_bounds": None,
                "is_skipped": True,
                "skip_reason": skip_reason
            }
            continue

        # 2. Extract Dimensions on this page for Local Calibration
        dim_candidates = extract_numeric_dimensions(page)

        # 3. Dynamic Legend Extraction on this page
        drawings = page.get_drawings()
        legend_info = extract_page_legend_and_swatches(page, drawings)
        page_palette = legend_info["dynamic_palette"]
        page_cutouts = legend_info["cutout_colors"]
        legend_bounds = legend_info["legend_bounds"]

        # Register newly discovered materials from this page's Legend into the global palette
        for mat_name, hex_list in page_palette.items():
            if mat_name not in global_palette:
                global_palette[mat_name] = []
            for hx in hex_list:
                if hx not in global_palette[mat_name]:
                    global_palette[mat_name].append(hx)

        for chx in page_cutouts:
            if chx not in global_cutouts:
                global_cutouts.append(chx)

        # 4. Extract Drawings & Classify Materials across full elevation
        page_paths: List[VectorPath] = []
        materials_dict: Dict[str, List[VectorPath]] = {}
        cutout_paths: List[VectorPath] = []

        path_counter = 1
        for d in drawings:
            stroke = d.get("color")
            fill = d.get("fill")
            width = d.get("width", 1.0)
            layer = d.get("layer")
            d_rect = d.get("rect", (0, 0, 0, 0))

            # Skip vector swatches inside the Legend container
            if legend_bounds:
                lx0, ly0, lx1, ly1 = legend_bounds
                if lx0 - 4 <= d_rect[0] and d_rect[2] <= lx1 + 4 and ly0 - 4 <= d_rect[1] and d_rect[3] <= ly1 + 4:
                    continue

            stroke_h = rgb_to_hex(stroke)
            fill_h = rgb_to_hex(fill)

            items = d.get("items", [])
            pts = []
            is_closed = d.get("closePath", False)
            cmd_type = "lines"

            for it in items:
                cmd = it[0]
                if cmd == "l":
                    p1, p2 = it[1], it[2]
                    if not pts:
                        pts.append((p1.x, p1.y))
                    pts.append((p2.x, p2.y))
                elif cmd == "re":
                    r = it[1]
                    pts = [
                        (r.x0, r.y0),
                        (r.x1, r.y0),
                        (r.x1, r.y1),
                        (r.x0, r.y1),
                        (r.x0, r.y0)
                    ]
                    is_closed = True
                    cmd_type = "rect"
                elif cmd == "c":
                    p1, p2, p3, p4 = it[1], it[2], it[3], it[4]
                    if not pts:
                        pts.append((p1.x, p1.y))
                    for t_step in [0.33, 0.66, 1.0]:
                        inv = 1.0 - t_step
                        bx = (inv**3)*p1.x + 3*(inv**2)*t_step*p2.x + 3*inv*(t_step**2)*p3.x + (t_step**3)*p4.x
                        by = (inv**3)*p1.y + 3*(inv**2)*t_step*p2.y + 3*inv*(t_step**2)*p3.y + (t_step**3)*p4.y
                        pts.append((bx, by))
                    cmd_type = "curve"

            if pts:
                if len(pts) >= 3 and math.isclose(pts[0][0], pts[-1][0], abs_tol=1e-3) and math.isclose(pts[0][1], pts[-1][1], abs_tol=1e-3):
                    is_closed = True

                bbox_tuple = (d_rect[0], d_rect[1], d_rect[2], d_rect[3]) if hasattr(d_rect, "__getitem__") else (0, 0, 0, 0)

                # Classify material using cumulative dynamic palette and CAD color distance tolerance
                mat_type, is_cutout = classify_vector_material(
                    stroke_h, fill_h,
                    palette_map=global_palette,
                    cutout_colors=global_cutouts
                )

                # Local Scale Calibration via nearby Dimension Text
                loc_scale, sc_src, d_w, d_h = calibrate_panel_local_scale_directional(
                    bbox_tuple, dim_candidates, scale_factor
                )

                v_path = VectorPath(
                    id=path_counter,
                    command_type=cmd_type,
                    points=pts,
                    is_closed=is_closed,
                    stroke_color=stroke[:3] if stroke else None,
                    fill_color=fill[:3] if fill else None,
                    stroke_width=width,
                    bbox=bbox_tuple,
                    layer=layer,
                    classified_material=mat_type,
                    local_scale=loc_scale,
                    scale_source=sc_src,
                    dim_w_mm=d_w,
                    dim_h_mm=d_h
                )
                path_counter += 1
                page_paths.append(v_path)

                if is_cutout:
                    cutout_paths.append(v_path)
                elif mat_type:
                    materials_dict.setdefault(mat_type, []).append(v_path)
                    all_materials_encountered.add(mat_type)
                    total_acoustic_paths_found += 1

        # Optional preview rendering
        preview_img = None
        if render_previews:
            try:
                pix = page.get_pixmap(dpi=preview_dpi)
                preview_img = Image.open(io.BytesIO(pix.tobytes("png")))
            except Exception:
                preview_img = None

        pages_result[page_num] = {
            "page_number": page_num,
            "page_index": page_idx,
            "sheet_name": sheet_name,
            "detected_scales": detected_scales,
            "width_pt": width_pt,
            "height_pt": height_pt,
            "materials": materials_dict,
            "cutout_paths": cutout_paths,
            "all_paths": page_paths,
            "total_material_paths": sum(len(p_list) for p_list in materials_dict.values()),
            "preview_image": preview_img,
            "dynamic_palette": page_palette,
            "cutout_colors": page_cutouts,
            "legend_bounds": legend_bounds,
            "is_skipped": False,
            "skip_reason": ""
        }

    doc.close()

    return {
        "document_name": doc_name,
        "total_pages": total_pages,
        "scale_factor": scale_factor,
        "pages": pages_result,
        "materials_found": sorted(list(all_materials_encountered)),
        "total_acoustic_paths": total_acoustic_paths_found
    }


def generate_sample_contractor_drawing(
    output_path: Union[str, Path],
    include_drafting_errors: bool = True,
    num_pages: int = 3
) -> str:
    """
    Generate a realistic multi-page architectural acoustic takeoff sample drawing (PDF)
    containing:
    - Page 1: Orion Innovation Hub - Level 4 Boardroom & Studio (Polyester 9mm & Wood Slat Panel)
    - Page 2: Project Cover Sheet & General Specifications (NO acoustic materials -> tests graceful skip!)
    - Page 3: Orion Hub - Level 5 Auditorium & Hall (Polyester 9mm, Wood Slat & Fabric Wrapped 25mm)
    """
    if fitz is None:
        raise ImportError("PyMuPDF (`fitz`) is required to generate sample PDF.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()

    # ==========================================
    # PAGE 1: LEVEL 4 FITOUT (BOARDROOM & STUDIO)
    # ==========================================
    page1 = doc.new_page(width=842, height=595)

    # Border
    page1.draw_rect(fitz.Rect(20, 20, 822, 575), color=(0.1, 0.1, 0.1), width=1.5)

    # Title block in bottom right
    page1.draw_rect(fitz.Rect(540, 480, 815, 565), color=(0.2, 0.2, 0.2), fill=(0.95, 0.95, 0.95), width=1.0)
    page1.insert_text((550, 500), "PROJECT: ORION INNOVATION HUB", fontsize=9, fontname="helv", color=(0.1, 0.1, 0.1))
    page1.insert_text((550, 515), "DRAWING: LEVEL 4 ACOUSTIC WALL & CEILING FITOUT", fontsize=8, fontname="helv", color=(0.1, 0.1, 0.1))
    page1.insert_text((550, 530), "DWG NO: AC-401   |   SHEET: 1 OF 3", fontsize=8, fontname="helv", color=(0.1, 0.1, 0.1))
    page1.insert_text((550, 545), "SCALE: 1:50   |   UNITS: MILLIMETERS", fontsize=8, fontname="helv", color=(0.8, 0.1, 0.1))
    page1.insert_text((550, 558), "DATE: 2026-10-22   |   DRAWN BY: QS/AEC TEAM", fontsize=7, fontname="helv", color=(0.3, 0.3, 0.3))

    # Graphic Scale Bar
    scale_bar_x = 350
    scale_bar_y = 540
    page1.insert_text((scale_bar_x, scale_bar_y - 8), "GRAPHIC SCALE 1:50 (METERS)", fontsize=7, fontname="helv")
    seg_w = 56.7
    for i in range(3):
        r_box = fitz.Rect(scale_bar_x + i * seg_w, scale_bar_y, scale_bar_x + (i + 1) * seg_w, scale_bar_y + 6)
        fill_col = (0.1, 0.1, 0.1) if i % 2 == 0 else (1, 1, 1)
        page1.draw_rect(r_box, color=(0, 0, 0), fill=fill_col, width=0.75)
        page1.insert_text((scale_bar_x + i * seg_w - 2, scale_bar_y + 16), f"{i}m", fontsize=6, fontname="helv")
    page1.insert_text((scale_bar_x + 3 * seg_w - 2, scale_bar_y + 16), "3m", fontsize=6, fontname="helv")

    # Legend - Bilingual AEC Specifications with Grey & Brown acoustic finishes
    page1.draw_rect(fitz.Rect(30, 460, 315, 568), color=(0.4, 0.4, 0.4), fill=(0.97, 0.97, 0.97), width=0.75)
    page1.insert_text((40, 475), "BẢNG GHI CHÚ - ACOUSTIC SPECIFICATIONS", fontsize=8, fontname="hebo", color=(0, 0, 0))
    # Item 1: Polyester 9mm (Cyan)
    page1.draw_rect(fitz.Rect(40, 485, 55, 497), color=(0.0, 0.47, 0.85), fill=(0.85, 0.92, 0.98), width=1.0)
    page1.insert_text((62, 494), "Type AP-1: Polyester 9mm Acoustic Panel", fontsize=7, fontname="helv")
    # Item 2: Wood Slat (Amber)
    page1.draw_rect(fitz.Rect(40, 502, 55, 514), color=(0.85, 0.50, 0.10), fill=(1.0, 0.95, 0.85), width=1.0)
    page1.insert_text((62, 511), "Type AC-2: Wood Slat Acoustic Ceiling Panel", fontsize=7, fontname="helv")
    # Item 3: TP605: Tấm Tiêu Âm Sợi Polyester Xám Ghi (Grey #808080)
    page1.draw_rect(fitz.Rect(40, 519, 55, 531), color=(0.35, 0.35, 0.35), fill=(0.502, 0.502, 0.502), width=1.0)
    page1.insert_text((62, 528), "TIÊU ÂM TP605: Tấm Tiêu Âm Sợi Polyester Xám", fontsize=7, fontname="helv")
    # Item 4: TP603: Gỗ Tiêu Âm Đục Lỗ Nâu Óc Chó (Brown #8B4513)
    page1.draw_rect(fitz.Rect(40, 536, 55, 548), color=(0.3, 0.15, 0.05), fill=(0.545, 0.271, 0.075), width=1.0)
    page1.insert_text((62, 545), "GỖ TIÊU ÂM TP603: Gỗ Đục Lỗ Melamine Nâu", fontsize=7, fontname="helv")
    # Item 5: Door / Window Cutouts (Red)
    page1.draw_line(fitz.Point(40, 557), fitz.Point(55, 557), color=(0.85, 0.15, 0.15), width=1.2, dashes="[3 2]")
    page1.insert_text((62, 560), "Wall Cutouts / Glazing Openings (Deductions)", fontsize=7, fontname="helv")

    # Room 1: Executive Boardroom 101
    page1.draw_rect(fitz.Rect(50, 50, 400, 450), color=(0.1, 0.1, 0.1), width=2.5)
    page1.insert_text((60, 75), "EXECUTIVE BOARDROOM 101", fontsize=11, fontname="helv", color=(0.1, 0.1, 0.1))
    page1.insert_text((60, 90), "Area: 35.0 m² | Acoustic Class: STC 52", fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))

    # Room 1 Panels (Polyester 9mm)
    for i in range(4):
        p_rect = fitz.Rect(70 + i * 75, 55, 140 + i * 75, 100)
        page1.draw_rect(p_rect, color=(0.0, 0.47, 0.85), fill=(0.85, 0.92, 0.98), width=1.2)
        page1.insert_text((75 + i * 75, 80), f"POLY-1.{i+1}", fontsize=6.5, fontname="helv", color=(0.0, 0.3, 0.6))

    for i in range(3):
        p_rect = fitz.Rect(55, 120 + i * 95, 105, 205 + i * 95)
        page1.draw_rect(p_rect, color=(0.0, 0.47, 0.85), fill=(0.85, 0.92, 0.98), width=1.2)
        page1.insert_text((60, 165 + i * 95), f"POLY-W{i+1}", fontsize=6.5, fontname="helv", color=(0.0, 0.3, 0.6))

    # South Wall with door cutout
    page1.draw_rect(fitz.Rect(70, 405, 200, 445), color=(0.0, 0.47, 0.85), fill=(0.85, 0.92, 0.98), width=1.2)
    page1.insert_text((115, 428), "POLY-S1", fontsize=7, fontname="helv", color=(0.0, 0.3, 0.6))
    page1.draw_rect(fitz.Rect(205, 405, 255, 445), color=(0.85, 0.15, 0.15), width=1.0, dashes="[3 2]")
    page1.insert_text((210, 428), "DOOR VOID", fontsize=6, fontname="helv", color=(0.8, 0.1, 0.1))
    page1.draw_rect(fitz.Rect(260, 405, 380, 445), color=(0.0, 0.47, 0.85), fill=(0.85, 0.92, 0.98), width=1.2)
    page1.insert_text((300, 428), "POLY-S2", fontsize=7, fontname="helv", color=(0.0, 0.3, 0.6))

    # Boardroom Wood Slat Ceiling Clouds (Wood Slat Panel)
    cloud_1 = fitz.Rect(140, 140, 360, 230)
    page1.draw_rect(cloud_1, color=(0.85, 0.50, 0.10), fill=(1.0, 0.95, 0.85), width=1.2)
    page1.insert_text((190, 190), "WOOD SLAT CLOUD WS-01 (3.8m x 1.6m)", fontsize=7.5, fontname="helv", color=(0.6, 0.3, 0.0))

    cloud_2 = fitz.Rect(140, 250, 360, 360)
    page1.draw_rect(cloud_2, color=(0.85, 0.50, 0.10), fill=(1.0, 0.95, 0.85), width=1.2)
    page1.insert_text((190, 310), "WOOD SLAT CLOUD WS-02 (3.8m x 1.9m)", fontsize=7.5, fontname="helv", color=(0.6, 0.3, 0.0))
    # Light Cutout in Wood Slat Cloud
    light_cutout = fitz.Rect(220, 290, 280, 320)
    page1.draw_rect(light_cutout, color=(0.85, 0.15, 0.15), width=1.0, dashes="[2 2]")
    page1.insert_text((226, 307), "LIGHT CUTOUT", fontsize=5.5, fontname="helv", color=(0.8, 0.1, 0.1))

    # Room 2: Video Conference Studio 102
    page1.draw_rect(fitz.Rect(440, 50, 790, 450), color=(0.1, 0.1, 0.1), width=2.5)
    page1.insert_text((455, 75), "MEDIA & RECORDING STUDIO 102", fontsize=11, fontname="helv", color=(0.1, 0.1, 0.1))
    page1.insert_text((455, 90), "Area: 32.0 m² | Acoustic Treatment: Full Perimeter", fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))

    for j in range(4):
        y_top = 110 + j * 68
        p_box = fitz.Rect(730, y_top, 785, y_top + 55)
        page1.draw_rect(p_box, color=(0.0, 0.47, 0.85), fill=(0.85, 0.92, 0.98), width=1.2)
        page1.insert_text((735, y_top + 32), f"POLY-E{j+1}", fontsize=6.5, fontname="helv", color=(0.0, 0.3, 0.6))

    # Wood slat baffles in Studio 102
    for r in range(3):
        for c in range(2):
            bx = 480 + c * 90
            by = 140 + r * 75
            baffle_rect = fitz.Rect(bx, by, bx + 70, by + 50)
            page1.draw_rect(baffle_rect, color=(0.85, 0.50, 0.10), fill=(1.0, 0.95, 0.85), width=1.1)
            page1.insert_text((bx + 8, by + 28), f"WOOD-BF-{r+1}{c+1}", fontsize=6.5, fontname="helv", color=(0.55, 0.25, 0.0))

    # Studio 102 Grey PET Panels (TIÊU ÂM TP605 - CAD shift #828282)
    for g_idx in range(2):
        gy_top = 220 + g_idx * 75
        g_box = fitz.Rect(655, gy_top, 715, gy_top + 60)
        page1.draw_rect(g_box, color=(0.35, 0.35, 0.35), fill=(0.51, 0.51, 0.51), width=1.1)
        page1.insert_text((660, gy_top + 35), f"TP605-G{g_idx+1}", fontsize=6.5, fontname="helv", color=(1, 1, 1))

    # Studio 102 Walnut Slat Feature (GỖ TIÊU ÂM TP603 - CAD shift #8A4410)
    for w_idx in range(2):
        wy_top = 370 + w_idx * 35
        w_box = fitz.Rect(540, wy_top, 640, wy_top + 28)
        page1.draw_rect(w_box, color=(0.3, 0.15, 0.05), fill=(0.54, 0.27, 0.06), width=1.1)
        page1.insert_text((550, wy_top + 18), f"TP603-W{w_idx+1}", fontsize=6.5, fontname="helv", color=(1, 1, 1))

    if include_drafting_errors:
        # Defect 1: Unclosed Geometry
        gap_pts = [
            fitz.Point(455, 380),
            fitz.Point(520, 380),
            fitz.Point(520, 435),
            fitz.Point(472, 435)
        ]
        for idx_p in range(len(gap_pts) - 1):
            page1.draw_line(gap_pts[idx_p], gap_pts[idx_p + 1], color=(0.0, 0.47, 0.85), width=1.3)
        page1.insert_text((460, 405), "POLY-ERR-GAP", fontsize=6.5, fontname="helv", color=(0.0, 0.3, 0.6))

        # Defect 2: Fake Dimension Override
        dim_x = 795
        page1.draw_line(fitz.Point(dim_x, 110), fitz.Point(dim_x + 12, 110), color=(0.2, 0.2, 0.2), width=0.7)
        page1.draw_line(fitz.Point(dim_x, 165), fitz.Point(dim_x + 12, 165), color=(0.2, 0.2, 0.2), width=0.7)
        page1.draw_line(fitz.Point(dim_x + 8, 110), fitz.Point(dim_x + 8, 165), color=(0.2, 0.2, 0.2), width=0.7)
        page1.insert_text((dim_x + 12, 142), "2400 mm", fontsize=7.0, fontname="hebo", color=(0.1, 0.1, 0.1))

        # Defect 3: Overlapping Duplicate Geometry
        dup_rect = fitz.Rect(480, 140, 480 + 70, 140 + 50)
        page1.draw_rect(dup_rect, color=(0.85, 0.50, 0.10), fill=(1.0, 0.95, 0.85), width=1.1)

        # Defect 4: Conflicting Scale Detail Note
        page1.insert_text((550, 465), "DETAIL CALLOUT SCALE: 1:100", fontsize=6.5, fontname="heit", color=(0.5, 0.5, 0.5))

    # ==============================================================
    # PAGE 2: GENERAL NOTES & COVER SHEET (NO ACOUSTIC MATERIALS)
    # Tests graceful skip of pages without acoustic linework!
    # ==============================================================
    if num_pages >= 2:
        page2 = doc.new_page(width=842, height=595)
        page2.draw_rect(fitz.Rect(20, 20, 822, 575), color=(0.1, 0.1, 0.1), width=1.5)

        # Title Block
        page2.draw_rect(fitz.Rect(540, 480, 815, 565), color=(0.2, 0.2, 0.2), fill=(0.95, 0.95, 0.95), width=1.0)
        page2.insert_text((550, 500), "PROJECT: ORION INNOVATION HUB", fontsize=9, fontname="helv", color=(0.1, 0.1, 0.1))
        page2.insert_text((550, 515), "DRAWING: GENERAL NOTES & COVER SPECIFICATIONS", fontsize=8, fontname="helv", color=(0.1, 0.1, 0.1))
        page2.insert_text((550, 530), "DWG NO: AC-001   |   SHEET: 2 OF 3", fontsize=8, fontname="helv", color=(0.1, 0.1, 0.1))
        page2.insert_text((550, 545), "SCALE: N.T.S.   |   UNITS: MILLIMETERS", fontsize=8, fontname="helv", color=(0.3, 0.3, 0.3))
        page2.insert_text((550, 558), "DATE: 2026-10-22   |   DRAWN BY: QS/AEC TEAM", fontsize=7, fontname="helv", color=(0.3, 0.3, 0.3))

        # Main Cover Text
        page2.insert_text((60, 80), "ORION INNOVATION HUB — ACOUSTIC FITOUT PACKAGE", fontsize=16, fontname="hebo", color=(0.1, 0.1, 0.1))
        page2.insert_text((60, 105), "GENERAL ARCHITECTURAL FITOUT SPECIFICATIONS & SCHEDULE OF FINISHES", fontsize=10, fontname="helv", color=(0.3, 0.3, 0.3))

        notes = [
            "1.0 GENERAL CONTRACT CONDITIONS:",
            "  1.1 All dimensions shall be verified on site by the trade contractor prior to shop fabrication.",
            "  1.2 Do not scale drawings. Architectural dimensions take precedence over CAD vector linework.",
            "  1.3 All acoustic panels to meet ASTM E84 Class A flame spread index and NRC >= 0.85 rating.",
            "",
            "2.0 MATERIAL SPECIFICATIONS:",
            "  2.1 Polyester 9mm: 100% PET felt, minimum 60% post-consumer recycled fiber, density 1900 g/m².",
            "  2.2 Wood Slat Panel: Natural oak/walnut veneer on 9mm black acoustic polyester felt backing.",
            "  2.3 Fabric Wrapped 25mm: High-density 96 kg/m³ resin bonded fiberglass core with acoustically transparent textile.",
            "",
            "3.0 FABRICATION, TOLERANCE & CUTTING:",
            "  3.1 Standard panel stock sizes: 2400 x 1200 mm and 2440 x 1220 mm.",
            "  3.2 Cutting kerf allowance: 3.0 mm for CNC router / panel saw blades.",
            "  3.3 Perimeter shadow gap joint allowance: 5.0 mm unless noted otherwise on architectural detail drawings.",
            "",
            "NOTICE TO ESTIMATOR: THIS PAGE CONTAINS TEXT SPECIFICATIONS ONLY (NO ACOUSTIC TAKEOFF VECTOR POLYLINES)."
        ]
        curr_y = 150
        for n in notes:
            if n.startswith("NOTICE"):
                page2.insert_text((60, curr_y), n, fontsize=8.5, fontname="hebo", color=(0.7, 0.2, 0.2))
            elif n.startswith(("1.0", "2.0", "3.0")):
                page2.insert_text((60, curr_y), n, fontsize=9.5, fontname="hebo", color=(0.1, 0.1, 0.1))
            else:
                page2.insert_text((60, curr_y), n, fontsize=8.5, fontname="helv", color=(0.2, 0.2, 0.2))
            curr_y += 18

    # ==============================================================
    # PAGE 3: LEVEL 5 AUDITORIUM & LIVE PERFORMANCE HALL 201
    # Multi-material page (Polyester 9mm, Wood Slat, Fabric Wrapped)
    # ==============================================================
    if num_pages >= 3:
        page3 = doc.new_page(width=842, height=595)
        page3.draw_rect(fitz.Rect(20, 20, 822, 575), color=(0.1, 0.1, 0.1), width=1.5)

        # Title Block
        page3.draw_rect(fitz.Rect(540, 480, 815, 565), color=(0.2, 0.2, 0.2), fill=(0.95, 0.95, 0.95), width=1.0)
        page3.insert_text((550, 500), "PROJECT: ORION INNOVATION HUB", fontsize=9, fontname="helv", color=(0.1, 0.1, 0.1))
        page3.insert_text((550, 515), "DRAWING: LEVEL 5 AUDITORIUM ACOUSTIC TREATMENT", fontsize=8, fontname="helv", color=(0.1, 0.1, 0.1))
        page3.insert_text((550, 530), "DWG NO: AC-501   |   SHEET: 3 OF 3", fontsize=8, fontname="helv", color=(0.1, 0.1, 0.1))
        page3.insert_text((550, 545), "SCALE: 1:50   |   UNITS: MILLIMETERS", fontsize=8, fontname="helv", color=(0.8, 0.1, 0.1))
        page3.insert_text((550, 558), "DATE: 2026-10-22   |   DRAWN BY: QS/AEC TEAM", fontsize=7, fontname="helv", color=(0.3, 0.3, 0.3))

        # Auditorium Header
        page3.draw_rect(fitz.Rect(50, 50, 790, 460), color=(0.1, 0.1, 0.1), width=2.5)
        page3.insert_text((65, 78), "AUDITORIUM & MULTIPURPOSE HALL 201", fontsize=12, fontname="hebo", color=(0.1, 0.1, 0.1))
        page3.insert_text((65, 94), "Seating Capacity: 180 Persons | Target RT60: 0.65s | High Absorption & Diffusion Fitout", fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))

        # 1. Rear Wall: 6 Panels of Polyester 9mm (#0078D4)
        page3.insert_text((65, 120), "REAR ACOUSTIC WALL — POLYESTER 9MM PANELS", fontsize=8, fontname="hebo", color=(0.0, 0.3, 0.6))
        for p_idx in range(6):
            rx0 = 65 + p_idx * 68
            p_box = fitz.Rect(rx0, 130, rx0 + 60, 210)
            page3.draw_rect(p_box, color=(0.0, 0.47, 0.85), fill=(0.85, 0.92, 0.98), width=1.2)
            page3.insert_text((rx0 + 10, 175), f"POLY-A{p_idx+1}", fontsize=6.5, fontname="helv", color=(0.0, 0.3, 0.6))

        # Emergency Exit Door Cutout across panels 3 and 4
        page3.draw_rect(fitz.Rect(235, 170, 305, 210), color=(0.85, 0.15, 0.15), width=1.2, dashes="[3 2]")
        page3.insert_text((240, 192), "EXIT VOID", fontsize=6, fontname="helv", color=(0.8, 0.1, 0.1))

        # 2. Side Walls: 6 Large Wood Slat Panels (#D9822B)
        page3.insert_text((65, 240), "SIDE REFLECTOR & DIFFUSER WALLS — WOOD SLAT PANELS", fontsize=8, fontname="hebo", color=(0.6, 0.3, 0.0))
        for w_idx in range(6):
            wx0 = 65 + w_idx * 78
            w_box = fitz.Rect(wx0, 250, wx0 + 70, 335)
            page3.draw_rect(w_box, color=(0.85, 0.50, 0.10), fill=(1.0, 0.95, 0.85), width=1.2)
            page3.insert_text((wx0 + 8, 296), f"WOOD-AUD-{w_idx+1}", fontsize=6.5, fontname="helv", color=(0.55, 0.25, 0.0))

        # 3. Suspended Baffles: 4 Emerald Green Fabric Wrapped Baffles (#059669)
        page3.insert_text((65, 360), "OVERHEAD CANOPY BAFFLES — FABRIC WRAPPED 25MM", fontsize=8, fontname="hebo", color=(0.05, 0.55, 0.35))
        for b_idx in range(4):
            bx0 = 65 + b_idx * 120
            b_box = fitz.Rect(bx0, 375, bx0 + 105, 440)
            page3.draw_rect(b_box, color=(0.05, 0.59, 0.41), fill=(0.85, 0.96, 0.90), width=1.2)
            page3.insert_text((bx0 + 12, 412), f"FABRIC-BF-{b_idx+1}", fontsize=7.0, fontname="helv", color=(0.02, 0.45, 0.28))

    doc.save(str(output_path))
    doc.close()
    return str(output_path)
