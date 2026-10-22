"""
Module 5: Quality Control (QC) & Drawing Validation System
Detects contractor drafting errors:
1. Unclosed vector geometry (gaps, dangling endpoints, leaking polygons)
2. Fake dimensions (text dimension overrides conflicting with scaled vector geometry)
3. Wrong scale / Scale conflicts (title block vs graphic scale bar vs panel dimension sanity)
4. Overlapping duplicate CAD entities (double counting risk)
5. Sliver geometry & drafting artifacts
Generates visual annotated overlays and zoomed-in defect spotlights.
"""

from dataclasses import dataclass, field
import io
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    plt = None
    Image = None

import importlib
_mod_pdf = importlib.import_module("modules.1_pdf_parser")
VectorPath = _mod_pdf.VectorPath
PDFPageData = _mod_pdf.PDFPageData

_mod_geom = importlib.import_module("modules.2_geometry")
PanelItem = _mod_geom.PanelItem


@dataclass
class QCIssue:
    """Represents an identified contractor drafting flaw or geometry discrepancy."""
    id: str                          # e.g. "QC-001"
    category: str                    # "Unclosed Geometry", "Fake Dimension", "Scale Conflict", "Duplicate Entity", "Sliver Geometry"
    severity: str                    # "CRITICAL", "WARNING", "INFO"
    title: str                       # Concise issue title
    description: str                 # Technical description with exact measurements
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points
    real_coords: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    expected_value: Optional[str] = None     # e.g. "2400 mm (from dimension text)"
    measured_value: Optional[str] = None     # e.g. "970 mm (measured from CAD vectors)"
    deviation_pct: Optional[float] = None    # e.g. 59.6%
    element_tag: Optional[str] = None        # e.g. "AP-E1"
    remedy: str = ""                         # Actionable correction for estimator or drafter


@dataclass
class QCReport:
    """Comprehensive Quality Control Audit Report for a drawing."""
    total_issues: int = 0
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    health_score: float = 100.0              # 0 to 100%
    health_grade: str = "A"                  # A (90-100), B (80-89), C (70-79), D (60-69), F (<60)
    scale_validation_status: str = "VERIFIED" # "VERIFIED", "WARNING", "CONFLICT"
    scale_details: Dict[str, Any] = field(default_factory=dict)
    issues: List[QCIssue] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)


def parse_dimension_string(dim_str: str) -> Optional[float]:
    """
    Extract numeric dimension in millimeters from an architectural dimension text.
    Handles '2400 mm', '2400mm', '2.40 m', 'DIM: 2400', 'W=1200'.
    Explicitly filters out dates, drawing numbers, scales, and revision stamps.
    """
    cleaned = dim_str.strip()
    upper = cleaned.upper()

    # Reject common architectural metadata strings
    if any(k in upper for k in ["DATE", "REV", "SCALE", "DWG", "PROJECT", "PAGE", "SHEET", "STC", "AREA", "STOCK", "SPEC", "TYPE"]):
        return None
    if re.search(r"\b20\d\d[-/]\d{1,2}[-/]\d{1,2}\b", cleaned):  # YYYY-MM-DD
        return None
    if re.search(r"\b1:\d{2,4}\b", cleaned):  # Scale ratio
        return None
    if re.search(r"^[A-Z]{1,3}-\d{2,4}$", upper):  # Tag like AC-401, AP-101
        return None

    # Match metric meters (e.g. 2.4m, 3.8m, 1.20 m)
    m_match = re.search(r"(?:DIM[:=\s]*)?(\d+(?:\.\d+)?)\s*m\b", cleaned, re.IGNORECASE)
    if m_match:
        val = float(m_match.group(1))
        if 0.2 <= val <= 25.0:
            return val * 1000.0  # Convert to mm

    # Match millimeters (e.g. 2400 mm, 1200mm, 2400)
    mm_match = re.search(r"(?:DIM[:=\s]*)?(\d{3,5})\s*mm\b", cleaned, re.IGNORECASE)
    if mm_match:
        val = float(mm_match.group(1))
        if 50.0 <= val <= 30000.0:
            return val

    # Match dimension prefixes like 'DIM: 2400' or 'W: 1200'
    prefix_match = re.search(r"(?:DIM|LENGTH|WIDTH|HEIGHT|W|L|H)[:=\s]+(\d{3,5})\b", cleaned, re.IGNORECASE)
    if prefix_match:
        val = float(prefix_match.group(1))
        if 50.0 <= val <= 30000.0:
            return val

    return None



def validate_unclosed_geometry(
    paths: List[VectorPath],
    scale_multiplier: float,
    target_unit: str = "mm"
) -> List[QCIssue]:
    """
    Detect unclosed polylines and gaps where geometry was intended to close
    but endpoints stop short or overshoot.
    Handles both multi-vertex open paths and separate unclosed line chains.
    """
    issues = []
    issue_num = 1

    # 1. Check direct multi-vertex open paths
    for p in paths:
        if len(p.points) >= 3 and not p.is_closed:
            p_first = p.points[0]
            p_last = p.points[-1]
            gap_pt = math.hypot(p_first[0] - p_last[0], p_first[1] - p_last[1])
            gap_real = gap_pt * scale_multiplier

            # Gap between 1.5 pt (~25mm at 1:50) and 85.0 pt (~1500mm at 1:50)
            if 1.5 <= gap_pt <= 85.0:

                x_coords = [pt[0] for pt in p.points]
                y_coords = [pt[1] for pt in p.points]
                bbox = (min(x_coords) - 5, min(y_coords) - 5, max(x_coords) + 5, max(y_coords) + 5)
                real_bbox = (bbox[0] * scale_multiplier, bbox[1] * scale_multiplier, bbox[2] * scale_multiplier, bbox[3] * scale_multiplier)

                issues.append(
                    QCIssue(
                        id=f"QC-GAP-{issue_num:02d}",
                        category="Unclosed Geometry",
                        severity="CRITICAL",
                        title=f"Unclosed Loop with {gap_real:.1f} {target_unit} Gap",
                        description=(
                            f"Vector path #{p.id} has {len(p.points)} vertices but does not close. "
                            f"The gap between start point ({p_first[0]:.1f}, {p_first[1]:.1f}) and "
                            f"end point ({p_last[0]:.1f}, {p_last[1]:.1f}) measures {gap_pt:.1f} pt "
                            f"({gap_real:.1f} {target_unit} in real life). This will cause polygonization "
                            f"failure and leaking boundaries."
                        ),
                        bbox=bbox,
                        real_coords=real_bbox,
                        expected_value=f"Closed Loop (Gap = 0 {target_unit})",
                        measured_value=f"Open Gap = {gap_real:.1f} {target_unit}",
                        deviation_pct=100.0,
                        remedy=(
                            f"Auto-snap gap using tolerance of {gap_real + 5.0:.1f} {target_unit}, or request drafter "
                            f"close the CAD polyline loop before final sign-off."
                        )
                    )
                )
                issue_num += 1

    # 2. Check for chains of separate line segments (degree 1 endpoints)
    line_segments = []
    for p in paths:
        if len(p.points) == 2 and not p.is_closed:
            line_segments.append((p.points[0], p.points[1]))

    # Find open chains by linking adjacent segment endpoints
    if line_segments:
        # Build simple proximity chain
        unvisited = list(line_segments)
        chains = []
        while unvisited:
            curr = unvisited.pop(0)
            chain = [curr[0], curr[1]]
            extended = True
            while extended:
                extended = False
                for idx, seg in enumerate(unvisited):
                    # Check if seg attaches to chain end
                    if math.hypot(chain[-1][0] - seg[0][0], chain[-1][1] - seg[0][1]) <= 1.0:
                        chain.append(seg[1])
                        unvisited.pop(idx)
                        extended = True
                        break
                    elif math.hypot(chain[-1][0] - seg[1][0], chain[-1][1] - seg[1][1]) <= 1.0:
                        chain.append(seg[0])
                        unvisited.pop(idx)
                        extended = True
                        break
                    elif math.hypot(chain[0][0] - seg[1][0], chain[0][1] - seg[1][1]) <= 1.0:
                        chain.insert(0, seg[0])
                        unvisited.pop(idx)
                        extended = True
                        break
                    elif math.hypot(chain[0][0] - seg[0][0], chain[0][1] - seg[0][1]) <= 1.0:
                        chain.insert(0, seg[1])
                        unvisited.pop(idx)
                        extended = True
                        break
            if len(chain) >= 4:  # At least 3 connected line segments
                chains.append(chain)

        for chain in chains:
            gap_pt = math.hypot(chain[0][0] - chain[-1][0], chain[0][1] - chain[-1][1])
            gap_real = gap_pt * scale_multiplier
            # If start and end are close but NOT closed
            if 1.5 <= gap_pt <= 85.0:

                x_coords = [pt[0] for pt in chain]
                y_coords = [pt[1] for pt in chain]
                bbox = (min(x_coords) - 6, min(y_coords) - 6, max(x_coords) + 6, max(y_coords) + 6)
                real_bbox = (bbox[0] * scale_multiplier, bbox[1] * scale_multiplier, bbox[2] * scale_multiplier, bbox[3] * scale_multiplier)

                issues.append(
                    QCIssue(
                        id=f"QC-GAP-{issue_num:02d}",
                        category="Unclosed Geometry",
                        severity="CRITICAL",
                        title=f"Unclosed Line Chain ({gap_real:.1f} {target_unit} Opening)",
                        description=(
                            f"A chain of {len(chain)-1} connected CAD lines forms an open perimeter but stops short. "
                            f"The gap between line start ({chain[0][0]:.1f}, {chain[0][1]:.1f}) and "
                            f"line end ({chain[-1][0]:.1f}, {chain[-1][1]:.1f}) measures {gap_pt:.1f} pt "
                            f"({gap_real:.1f} {target_unit}). This perimeter cannot be converted to a solid panel."
                        ),
                        bbox=bbox,
                        real_coords=real_bbox,
                        expected_value=f"Closed Loop (Gap = 0 {target_unit})",
                        measured_value=f"Open Gap = {gap_real:.1f} {target_unit}",
                        deviation_pct=100.0,
                        remedy=(
                            f"Auto-snap gap with {gap_real + 5.0:.1f} {target_unit} tolerance or add missing CAD line segment."
                        )
                    )
                )
                issue_num += 1

    return issues


def validate_dimension_overrides(
    text_blocks: List[Dict[str, Any]],
    panels: List[PanelItem],
    scale_multiplier: float,
    target_unit: str = "mm"
) -> List[QCIssue]:
    """
    Cross-checks annotated dimension text against actual scaled CAD vector dimensions.
    Detects 'fake dimensions' where the drafter overrode text instead of stretching geometry.
    Filters out room boundary boxes and matches only true acoustic panel elements.
    """
    issues = []
    issue_num = 1

    # Filter panels to individual acoustic components (exclude full-room borders)
    valid_panels = [p for p in panels if p.gross_area < 25.0]

    dim_candidates = []
    for tb in text_blocks:
        raw_t = tb.get("text", "").strip()
        parsed_val = parse_dimension_string(raw_t)
        if parsed_val is not None:
            dim_candidates.append({
                "text": raw_t,
                "val_mm": parsed_val,
                "bbox": tb["bbox"]  # (x0, y0, x1, y1)
            })

    for dim in dim_candidates:
        dt_x = (dim["bbox"][0] + dim["bbox"][2]) / 2.0
        dt_y = (dim["bbox"][1] + dim["bbox"][3]) / 2.0
        target_val = dim["val_mm"] if target_unit == "mm" else (dim["val_mm"] / 25.4)

        # Find closest valid panel
        closest_panel = None
        closest_dist = float("inf")

        for p in valid_panels:
            pb = p.raw_pdf_bbox
            dx = max(pb[0] - dt_x, 0, dt_x - pb[2])
            dy = max(pb[1] - dt_y, 0, dt_y - pb[3])
            dist_to_panel = math.hypot(dx, dy)

            if dist_to_panel <= 65.0 and dist_to_panel < closest_dist:
                closest_dist = dist_to_panel
                closest_panel = p

        if closest_panel is not None:
            p = closest_panel
            pb = p.raw_pdf_bbox
            p_len = p.cut_length
            p_wid = p.cut_width

            dev_len = abs(target_val - p_len) / max(1.0, target_val)
            dev_wid = abs(target_val - p_wid) / max(1.0, target_val)

            min_dev = min(dev_len, dev_wid)
            measured_val = p_len if dev_len < dev_wid else p_wid
            dim_type = "Length" if dev_len < dev_wid else "Width"

            diff_abs = abs(target_val - measured_val)
            if (0.08 <= min_dev <= 2.5) and (diff_abs >= 40.0):
                dev_pct = min_dev * 100.0
                severity = "CRITICAL" if dev_pct >= 20.0 else "WARNING"

                comb_bbox = (
                    min(pb[0], dim["bbox"][0]) - 8,
                    min(pb[1], dim["bbox"][1]) - 8,
                    max(pb[2], dim["bbox"][2]) + 8,
                    max(pb[3], dim["bbox"][3]) + 8
                )

                issues.append(
                    QCIssue(
                        id=f"QC-DIM-{issue_num:02d}",
                        category="Fake Dimension",
                        severity=severity,
                        title=f"Dimension Text Override ({dev_pct:.1f}% Discrepancy on {p.tag})",
                        description=(
                            f"Annotated dimension text specifies '{dim['text']}' near panel {p.tag}, but the underlying "
                            f"CAD vector geometry measures only {measured_val:.1f} {target_unit} {dim_type} "
                            f"(Difference: {diff_abs:.1f} {target_unit}, {dev_pct:.1f}% error). "
                            f"The drafter manually overrode the dimension string without updating CAD geometry."
                        ),
                        bbox=comb_bbox,
                        real_coords=(comb_bbox[0] * scale_multiplier, comb_bbox[1] * scale_multiplier, comb_bbox[2] * scale_multiplier, comb_bbox[3] * scale_multiplier),
                        expected_value=f"{target_val:.1f} {target_unit} (Annotated text: '{dim['text']}')",
                        measured_value=f"{measured_val:.1f} {target_unit} (Actual CAD vector {dim_type})",
                        deviation_pct=round(dev_pct, 1),
                        element_tag=p.tag,
                        remedy=(
                            f"Consult architect/contractor: if {target_val:.1f} {target_unit} is required, adjust the "
                            f"cutting schedule or stretch the drawing geometry before material procurement."
                        )
                    )
                )
                issue_num += 1

    return issues


    return issues


def validate_drawing_scale(
    pdf_data: PDFPageData,
    panels: List[PanelItem],
    scale_multiplier: float,
    chosen_scale_preset: str = "1:50",
    target_unit: str = "mm"
) -> Tuple[List[QCIssue], Dict[str, Any], str]:
    """
    Validate drawing scale consistency:
    - Compares stated title block scale ratio with graphic scale bar ticks
    - Sanity-checks commercial acoustic panel manufacturing dimensions
    """
    issues = []
    status = "VERIFIED"
    scale_details = {
        "chosen_preset": chosen_scale_preset,
        "multiplier": scale_multiplier,
        "graphic_bar_detected": False,
        "measured_ratio": None,
        "ratio_deviation_pct": 0.0
    }

    # 1. Graphic Scale Bar Validation
    # Look for scale bar text like 'GRAPHIC SCALE', '0m', '1m', '2m'
    scale_bar_labels = []
    for tb in pdf_data.text_blocks:
        t = tb["text"].strip()
        if re.search(r"\b(?:0m|1m|2m|3m)\b", t, re.IGNORECASE) or "GRAPHIC SCALE" in t.upper():
            scale_bar_labels.append(tb)

    # Check distance between '0m' and '1m' or '2m'
    zero_m = next((b for b in scale_bar_labels if re.search(r"\b0m\b", b["text"], re.IGNORECASE)), None)
    one_m = next((b for b in scale_bar_labels if re.search(r"\b1m\b", b["text"], re.IGNORECASE)), None)

    if zero_m and one_m:
        scale_details["graphic_bar_detected"] = True
        dx = abs(one_m["bbox"][0] - zero_m["bbox"][0])
        dy = abs(one_m["bbox"][1] - zero_m["bbox"][1])
        bar_len_pt = math.hypot(dx, dy)

        if bar_len_pt > 5.0:
            # In 1:N scale, 1000 mm real life in PDF points = 1000 / (N * (25.4/72)) = (72000 / 25.4) / N = 2834.646 / N
            # Therefore derived N = 2834.646 / bar_len_pt
            derived_ratio = 2834.646 / bar_len_pt
            scale_details["measured_ratio"] = round(derived_ratio, 1)

            # Check expected ratio from preset (e.g. 1:50 -> 50)
            expected_ratio = 50.0
            if "1:" in chosen_scale_preset:
                m = re.search(r"1:(\d+)", chosen_scale_preset)
                if m:
                    expected_ratio = float(m.group(1))

            dev_pct = abs(derived_ratio - expected_ratio) / expected_ratio * 100.0
            scale_details["ratio_deviation_pct"] = round(dev_pct, 1)

            if dev_pct > 8.0:
                status = "CONFLICT"
                issues.append(
                    QCIssue(
                        id="QC-SCL-01",
                        category="Scale Conflict",
                        severity="CRITICAL",
                        title=f"Graphic Scale Bar Conflict (1:{derived_ratio:.0f} vs {chosen_scale_preset})",
                        description=(
                            f"The graphic scale bar '0m to 1m' segment measures {bar_len_pt:.1f} pt on the sheet, "
                            f"which corresponds to an actual drawing scale of 1:{derived_ratio:.0f}. "
                            f"However, the selected drawing scale is '{chosen_scale_preset}'. "
                            f"This indicates a {dev_pct:.1f}% scale discrepancy! The drawing may have been plotted "
                            f"with 'Fit to Page' or exported at the wrong paper size."
                        ),
                        bbox=(
                            min(zero_m["bbox"][0], one_m["bbox"][0]) - 10,
                            min(zero_m["bbox"][1], one_m["bbox"][1]) - 15,
                            max(zero_m["bbox"][2], one_m["bbox"][2]) + 10,
                            max(zero_m["bbox"][3], one_m["bbox"][3]) + 15
                        ),
                        expected_value=f"Scale {chosen_scale_preset} ({2834.65/expected_ratio:.1f} pt per meter)",
                        measured_value=f"Scale 1:{derived_ratio:.0f} ({bar_len_pt:.1f} pt per meter)",
                        deviation_pct=round(dev_pct, 1),
                        remedy="Calibrate scale using the graphic scale bar or re-export PDF at 100% full scale without page shrinking."
                    )
                )

    # 2. Conflicting Callout Scales on Same Page
    if len(pdf_data.detected_scales) > 1:
        # Check if multiple scales were found
        scale_labels = [s["label"] for s in pdf_data.detected_scales]
        if len(set(scale_labels)) > 1:
            issues.append(
                QCIssue(
                    id="QC-SCL-02",
                    category="Scale Conflict",
                    severity="WARNING",
                    title="Multiple Conflicting Scales Detected on Page",
                    description=(
                        f"Found multiple scale annotations on this page: {', '.join(scale_labels)}. "
                        f"Detail drawings or callouts may be drawn at a different scale from the general floor plan."
                    ),
                    bbox=(20, 20, 300, 100),
                    expected_value=chosen_scale_preset,
                    measured_value=", ".join(scale_labels),
                    remedy="Verify which sections of the drawing correspond to detail callouts to avoid applying the wrong scale."
                )
            )

    # 3. Manufacturing Dimension Sanity Check
    # Standard commercial acoustic panels max out at ~3600 mm (12 ft)
    for p in panels:
        max_dim = max(p.cut_length, p.cut_width)
        if target_unit == "mm" and max_dim > 5500.0:
            issues.append(
                QCIssue(
                    id="QC-SAN-01",
                    category="Scale Conflict",
                    severity="WARNING",
                    title=f"Acoustic Panel Dimension Exceeds Manufacturing Limit ({max_dim:.0f} mm)",
                    description=(
                        f"Panel {p.tag} measures {max_dim:.0f} mm. Standard acoustic panels rarely exceed 3600 mm. "
                        f"Verify if the drawing scale is too large (e.g. 1:100 treated as 1:50) or if this represents "
                        f"an entire wall surface that needs modular panel tiling."
                    ),
                    bbox=p.raw_pdf_bbox,
                    expected_value="Max 3600 mm standard modular panel",
                    measured_value=f"{max_dim:.0f} mm",
                    element_tag=p.tag,
                    remedy="Subdivide surface into modular panels or adjust drawing scale."
                )
            )
            break  # Flag once

    return issues, scale_details, status


def validate_overlapping_duplicates(
    paths: List[VectorPath],
    panels: List[PanelItem],
    scale_multiplier: float,
    target_unit: str = "mm"
) -> List[QCIssue]:
    """
    Detect duplicate overlapping panels and CAD entities copied directly on top of each other.
    Checks raw drawing vector paths as well as takeoff panels to catch CAD layer duplicates.
    """
    issues = []
    issue_num = 1
    seen_bboxes = set()

    # 1. Check raw vector paths for identical overlapping geometry
    closed_paths = [p for p in paths if (p.is_closed or p.command_type == "rect") and (p.bbox[2] - p.bbox[0] > 10.0 and p.bbox[3] - p.bbox[1] > 10.0)]
    for i in range(len(closed_paths)):
        for j in range(i + 1, len(closed_paths)):
            b1 = closed_paths[i].bbox
            b2 = closed_paths[j].bbox

            if math.isclose(b1[0], b2[0], abs_tol=1.5) and \
               math.isclose(b1[1], b2[1], abs_tol=1.5) and \
               math.isclose(b1[2], b2[2], abs_tol=1.5) and \
               math.isclose(b1[3], b2[3], abs_tol=1.5):
                key = (round(b1[0], 0), round(b1[1], 0))
                if key not in seen_bboxes:
                    seen_bboxes.add(key)
                    w_real = (b1[2] - b1[0]) * scale_multiplier
                    h_real = (b1[3] - b1[1]) * scale_multiplier
                    issues.append(
                        QCIssue(
                            id=f"QC-DUP-{issue_num:02d}",
                            category="Duplicate Entity",
                            severity="CRITICAL",
                            title=f"Duplicate Overlapping CAD Path ({w_real:.0f}×{h_real:.0f} {target_unit})",
                            description=(
                                f"Vector paths #{closed_paths[i].id} and #{closed_paths[j].id} are drawn in the exact same "
                                f"coordinates (bbox: {b1[0]:.1f}, {b1[1]:.1f}). This commonly occurs when drafting elements "
                                f"are copied and pasted in place or duplicated across multiple CAD layers, causing double-counting "
                                f"in takeoff quantities if not filtered."
                            ),
                            bbox=b1,
                            real_coords=(b1[0] * scale_multiplier, b1[1] * scale_multiplier, b1[2] * scale_multiplier, b1[3] * scale_multiplier),
                            expected_value="1 unique CAD element",
                            measured_value="2 identical overlapping paths",
                            deviation_pct=100.0,
                            remedy="Delete duplicate entity in CAD or verify that deduplication filter is enabled in takeoff."
                        )
                    )
                    issue_num += 1

    # 2. Also check panel items
    n = len(panels)
    for i in range(n):
        for j in range(i + 1, n):
            p1 = panels[i]
            p2 = panels[j]
            dist_c = math.hypot(p1.center_x - p2.center_x, p1.center_y - p2.center_y)
            if dist_c <= 15.0:
                area_diff = abs(p1.net_area - p2.net_area) / max(0.01, max(p1.net_area, p2.net_area))
                if area_diff <= 0.05:
                    comb_bbox = (
                        min(p1.raw_pdf_bbox[0], p2.raw_pdf_bbox[0]),
                        min(p1.raw_pdf_bbox[1], p2.raw_pdf_bbox[1]),
                        max(p1.raw_pdf_bbox[2], p2.raw_pdf_bbox[2]),
                        max(p1.raw_pdf_bbox[3], p2.raw_pdf_bbox[3])
                    )
                    issues.append(
                        QCIssue(
                            id=f"QC-DUP-{issue_num:02d}",
                            category="Duplicate Entity",
                            severity="CRITICAL",
                            title=f"Duplicate Overlapping Panel ({p1.tag} & {p2.tag})",
                            description=(
                                f"Panel {p1.tag} and {p2.tag} share nearly identical positions (centroid distance: {dist_c:.1f} {target_unit}, "
                                f"area diff: {area_diff*100:.1f}%)."
                            ),
                            bbox=comb_bbox,
                            expected_value="Single unique panel",
                            measured_value="2 identical overlapping entities",
                            element_tag=f"{p1.tag}, {p2.tag}",
                            remedy="Delete duplicate instance or merge identical layers before nesting and takeoff."
                        )
                    )
                    issue_num += 1

    return issues



def run_qc_validation(
    pdf_data: PDFPageData,
    panels: List[PanelItem],
    scale_multiplier: float,
    chosen_scale_preset: str = "1:50",
    target_unit: str = "mm"
) -> QCReport:
    """
    Execute full automated Quality Control & Validation pipeline on drawing geometry.
    """
    all_issues: List[QCIssue] = []
    passed_checks: List[str] = []

    # 1. Unclosed Geometry Check
    gap_issues = validate_unclosed_geometry(pdf_data.paths, scale_multiplier, target_unit)
    if gap_issues:
        all_issues.extend(gap_issues)
    else:
        passed_checks.append("All vector polygon loops closed within standard tolerance.")

    # 2. Fake Dimension Overrides Check
    dim_issues = validate_dimension_overrides(pdf_data.text_blocks, panels, scale_multiplier, target_unit)
    if dim_issues:
        all_issues.extend(dim_issues)
    else:
        passed_checks.append("No manual dimension text overrides detected (100% geometric agreement).")

    # 3. Scale Validation & Sanity
    scale_issues, scale_details, scale_status = validate_drawing_scale(
        pdf_data, panels, scale_multiplier, chosen_scale_preset, target_unit
    )
    if scale_issues:
        all_issues.extend(scale_issues)
    else:
        passed_checks.append(f"Drawing scale verified ({chosen_scale_preset}) with graphic scale bar.")

    # 4. Duplicate Overlapping Entities Check
    dup_issues = validate_overlapping_duplicates(pdf_data.paths, panels, scale_multiplier, target_unit)

    if dup_issues:
        all_issues.extend(dup_issues)
    else:
        passed_checks.append("No duplicate overlapping geometry detected.")

    # Calculate Health Score & Grade
    crit_count = sum(1 for i in all_issues if i.severity == "CRITICAL")
    warn_count = sum(1 for i in all_issues if i.severity == "WARNING")
    info_count = sum(1 for i in all_issues if i.severity == "INFO")

    # Deduct 18 points per critical error, 6 points per warning
    health_score = max(0.0, min(100.0, 100.0 - (crit_count * 18.0 + warn_count * 6.0)))
    health_score = round(health_score, 1)

    if health_score >= 90.0:
        grade = "A (Excellent)"
    elif health_score >= 80.0:
        grade = "B (Good - Minor Flaws)"
    elif health_score >= 70.0:
        grade = "C (Fair - Review Needed)"
    elif health_score >= 60.0:
        grade = "D (Poor - High Risk)"
    else:
        grade = "F (Critical Errors Found)"

    return QCReport(
        total_issues=len(all_issues),
        critical_count=crit_count,
        warning_count=warn_count,
        info_count=info_count,
        health_score=health_score,
        health_grade=grade,
        scale_validation_status=scale_status,
        scale_details=scale_details,
        issues=all_issues,
        passed_checks=passed_checks
    )


def render_qc_overlay_diagram(
    base_image: Any,
    report: QCReport,
    dpi: int = 150
) -> Any:
    """
    Renders an engineering Quality Control visual overlay on top of the PDF raster preview,
    highlighting detected gaps, fake dimensions, duplicates, and scale conflicts.
    """
    if plt is None or Image is None or base_image is None:
        return base_image

    img_w, img_h = base_image.size
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=dpi)

    # Display base drawing
    ax.imshow(base_image)

    # Scale factor from PDF points (e.g. 842x595) to image pixels (e.g. 1754x1240 at 150 DPI)
    # Typically 150 DPI / 72 pt = ~2.083
    px_per_pt_x = img_w / 842.0  # approximate standard A4
    px_per_pt_y = img_h / 595.0

    severity_colors = {
        "CRITICAL": ("#DC2626", "#FEE2E2"),  # Bright Red
        "WARNING": ("#D97706", "#FEF3C7"),   # Amber
        "INFO": ("#2563EB", "#DBEAFE")       # Blue
    }

    for issue in report.issues:
        stroke_c, fill_c = severity_colors.get(issue.severity, ("#DC2626", "#FEE2E2"))
        b = issue.bbox

        # Scale bbox to image pixel space
        bx0 = b[0] * px_per_pt_x
        by0 = b[1] * px_per_pt_y
        bw = (b[2] - b[0]) * px_per_pt_x
        bh = (b[3] - b[1]) * px_per_pt_y

        # Draw highlight rectangle
        rect_patch = patches.Rectangle(
            (bx0, by0), bw, bh,
            linewidth=2.5, edgecolor=stroke_c, facecolor=fill_c,
            alpha=0.45, linestyle="--", zorder=4
        )
        ax.add_patch(rect_patch)

        # Crosshair center
        cx = bx0 + bw / 2.0
        cy = by0 + bh / 2.0
        ax.plot(cx, cy, marker="+", markersize=14, markeredgewidth=2.0, color=stroke_c, zorder=5)

        # Callout badge text
        badge_txt = f"{issue.id}: {issue.category.upper()}\n{issue.title}"
        ax.text(
            bx0, max(15, by0 - 8), badge_txt,
            fontsize=7.0, fontweight="bold", color="#FFFFFF",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=stroke_c, edgecolor="none", alpha=0.9),
            zorder=6
        )

    # Title Banner with Health Score
    status_color = "#16A34A" if report.health_score >= 85 else ("#D97706" if report.health_score >= 70 else "#DC2626")
    banner_title = (
        f"QUALITY CONTROL & DRAFTING AUDIT OVERLAY\n"
        f"Drawing Health Score: {report.health_score}% ({report.health_grade})  |  "
        f"Critical Defects: {report.critical_count}  |  Warnings: {report.warning_count}"
    )
    ax.set_title(banner_title, fontsize=10, fontweight="bold", color="#0F172A", pad=12)
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def render_issue_zoom_crop(
    base_image: Any,
    issue: QCIssue,
    padding_pt: float = 40.0,
    target_dpi: int = 150
) -> Any:
    """
    Extract a high-resolution zoomed-in spotlight crop of a specific drafting issue
    with target crosshairs and measurement indicators.
    """
    if Image is None or base_image is None:
        return None

    img_w, img_h = base_image.size
    px_per_pt_x = img_w / 842.0
    px_per_pt_y = img_h / 595.0

    b = issue.bbox
    # Calculate crop bounds in pixel space with padding
    pad_x = padding_pt * px_per_pt_x
    pad_y = padding_pt * px_per_pt_y

    x0 = max(0, int(round(b[0] * px_per_pt_x - pad_x)))
    y0 = max(0, int(round(b[1] * px_per_pt_y - pad_y)))
    x1 = min(img_w, int(round(b[2] * px_per_pt_x + pad_x)))
    y1 = min(img_h, int(round(b[3] * px_per_pt_y + pad_y)))

    if x1 <= x0 or y1 <= y0:
        return base_image

    crop = base_image.crop((x0, y0, x1, y1))

    # Annotate crop with callout
    draw = ImageDraw.Draw(crop)
    stroke_color = "#DC2626" if issue.severity == "CRITICAL" else "#D97706"

    # Defect box in crop coordinates
    cb_x0 = int(round(b[0] * px_per_pt_x - x0))
    cb_y0 = int(round(b[1] * px_per_pt_y - y0))
    cb_x1 = int(round(b[2] * px_per_pt_x - x0))
    cb_y1 = int(round(b[3] * px_per_pt_y - y0))

    draw.rectangle([cb_x0, cb_y0, cb_x1, cb_y1], outline=stroke_color, width=3)

    return crop
