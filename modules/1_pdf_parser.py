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


# Standard AEC Acoustic Material Color Palette Mappings
DEFAULT_MATERIAL_PALETTE: Dict[str, List[str]] = {
    "Polyester 9mm": [
        "#0078D4", "#0284C7", "#00AAFF", "#0047D4", "#005A9E", "#0077D4", "#38BDF8", "#0369A1"
    ],
    "Wood Slat Panel": [
        "#D9822B", "#D97706", "#F59E0B", "#B45309", "#92400E", "#D9821A", "#EA580C", "#CA8A04"
    ],
    "Fabric Wrapped 25mm": [
        "#059669", "#10B981", "#047857", "#065F46", "#34D399", "#15803D", "#16A34A"
    ]
}

# Cutout / Void Deduction Colors (Red, Crimson, Scarlet)
DEFAULT_CUTOUT_COLORS: List[str] = [
    "#D92626", "#EF4444", "#DC2626", "#FF0000", "#E11D48", "#B91C1C", "#991B1B", "#F87171"
]


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
    return dist / 441.67


def matches_color_list(candidate_hex: Optional[str], target_hexes: List[str], tolerance: float = 0.10) -> bool:
    """Check if candidate hex matches any color in target list within RGB tolerance."""
    if not candidate_hex or not target_hexes:
        return False
    for target in target_hexes:
        if color_distance(candidate_hex, target) <= tolerance:
            return True
    return False


def classify_vector_material(
    stroke_hex: Optional[str],
    fill_hex: Optional[str],
    palette_map: Optional[Dict[str, List[str]]] = None
) -> Tuple[Optional[str], bool]:
    """
    Classify acoustic material type based on line color, hatching, or fill.
    Returns: (material_type or None, is_cutout: bool)
    """
    palette = palette_map if palette_map is not None else DEFAULT_MATERIAL_PALETTE

    # 1. Check if it's a cutout / opening deduction
    for hex_val in (stroke_hex, fill_hex):
        if hex_val and matches_color_list(hex_val, DEFAULT_CUTOUT_COLORS, tolerance=0.10):
            return ("Cutout / Opening", True)

    # Filter out pure grays, whites, and dark border/wall lines
    test_hex = stroke_hex or fill_hex
    if not test_hex:
        return (None, False)

    r, g, b = hex_to_rgb(test_hex)
    max_c, min_c = max(r, g, b), min(r, g, b)
    is_grayscale = (max_c - min_c < 0.12) or (max_c < 0.28)
    if is_grayscale:
        # Ignore structural walls, grid lines, and title block borders
        return (None, False)

    # 2. Check explicitly mapped materials
    for mat_name, hex_list in palette.items():
        if stroke_hex and matches_color_list(stroke_hex, hex_list, tolerance=0.10):
            return (mat_name, False)
        if fill_hex and matches_color_list(fill_hex, hex_list, tolerance=0.10):
            return (mat_name, False)

    # 3. Heuristic Hue Fallback (Blue = Polyester 9mm, Orange = Wood Slat, Green = Fabric Wrapped)
    if b > r + 0.15 and b > g + 0.10:
        return ("Polyester 9mm", False)
    elif r > b + 0.20 and g > b + 0.08 and r >= g:
        return ("Wood Slat Panel", False)
    elif g > r + 0.15 and g > b + 0.12:
        return ("Fabric Wrapped 25mm", False)

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

    path_id = 1
    for d in drawings:
        stroke = d.get("color")
        fill = d.get("fill")
        width = d.get("width", 1.0)
        layer = d.get("layer")
        d_rect = d.get("rect", (0, 0, 0, 0))

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

            # Classify material
            mat_type, _ = classify_vector_material(stroke_h, fill_h)

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
                    classified_material=mat_type
                )
            )
            path_id += 1

    raw_text = page.get_text("text")
    blocks = page.get_text("blocks")
    text_data = []
    for b in blocks:
        if len(b) >= 5 and b[4].strip():
            text_data.append({
                "bbox": (b[0], b[1], b[2], b[3]),
                "text": b[4].strip()
            })

    detected_scales = detect_drawing_scales(raw_text)
    sheet_name = extract_sheet_or_room_name(page)

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
        sheet_name=sheet_name
    )


def scan_full_document(
    pdf_path: Union[str, Path, bytes, io.BytesIO],
    scale_factor: float = 17.6389,
    material_color_map: Optional[Dict[str, List[str]]] = None,
    render_previews: bool = False,
    preview_dpi: int = 100
) -> Dict[str, Any]:
    """
    Multi-Page Batch Scanner:
    1. Iterates through ALL pages in the PDF.
    2. Auto-detects Sheet Name or Room Name per page using `page.get_text("words")`.
    3. Classifies acoustic materials based on line colors, hatching, and fills
       (e.g., Blue lines = 'Polyester 9mm', Orange lines = 'Wood Slat Panel').
    4. Detects cutouts / opening deduction vectors.
    5. Returns a structured nested dictionary grouped by Page_Number and Material_Type.
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
    palette = material_color_map if material_color_map is not None else DEFAULT_MATERIAL_PALETTE

    pages_result: Dict[int, Dict[str, Any]] = {}
    all_materials_encountered = set()
    total_acoustic_paths_found = 0

    for page_idx in range(total_pages):
        page = doc[page_idx]
        rect = page.rect
        width_pt, height_pt = rect.width, rect.height
        page_num = page_idx + 1

        # 1. Auto-detect Sheet Name or Room Name via words
        sheet_name = extract_sheet_or_room_name(page)

        # 2. Extract Drawing Scale
        raw_text = page.get_text("text")
        detected_scales = detect_drawing_scales(raw_text)

        # 3. Extract Drawings & Classify Materials
        drawings = page.get_drawings()
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

                # Classify material
                mat_type, is_cutout = classify_vector_material(stroke_h, fill_h, palette)

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
                    classified_material=mat_type
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
            "preview_image": preview_img
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

    # Legend
    page1.draw_rect(fitz.Rect(30, 480, 310, 565), color=(0.4, 0.4, 0.4), fill=(0.97, 0.97, 0.97), width=0.75)
    page1.insert_text((40, 495), "LEGEND - ACOUSTIC SPECIFICATIONS", fontsize=8, fontname="helv", color=(0, 0, 0))
    # Item 1: Polyester 9mm (Cyan)
    page1.draw_rect(fitz.Rect(40, 505, 55, 517), color=(0.0, 0.47, 0.85), fill=(0.85, 0.92, 0.98), width=1.0)
    page1.insert_text((62, 514), "Type AP-1: Polyester 9mm Acoustic Panel", fontsize=7, fontname="helv")
    # Item 2: Wood Slat (Amber)
    page1.draw_rect(fitz.Rect(40, 523, 55, 535), color=(0.85, 0.50, 0.10), fill=(1.0, 0.95, 0.85), width=1.0)
    page1.insert_text((62, 532), "Type AC-2: Wood Slat Acoustic Ceiling Panel", fontsize=7, fontname="helv")
    # Item 3: Door / Window Cutouts (Red)
    page1.draw_line(fitz.Point(40, 545), fitz.Point(55, 545), color=(0.85, 0.15, 0.15), width=1.2, dashes="[3 2]")
    page1.insert_text((62, 548), "Wall Cutouts / Glazing Openings (Deductions)", fontsize=7, fontname="helv")

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
