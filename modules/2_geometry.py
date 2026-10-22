"""
Module 2: Geometry Processing & Takeoff Calculations
Reconstructs polygons from vector drawing paths, applies real-world scale factors,
detects and subtracts cutouts/voids page-by-page, computes net areas,
classifies acoustic materials, and extracts bounding fabrication dimensions.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from shapely.geometry import (
        GeometryCollection,
        LinearRing,
        LineString,
        MultiLineString,
        MultiPolygon,
        Point,
        Polygon,
        box
    )
    from shapely.ops import linemerge, polygonize, snap, unary_union
    from shapely.validation import make_valid
except ImportError:
    Polygon = None

import importlib

_mod_pdf = importlib.import_module("modules.1_pdf_parser")
VectorPath = _mod_pdf.VectorPath
color_distance = _mod_pdf.color_distance
classify_vector_material = _mod_pdf.classify_vector_material


@dataclass
class PanelItem:
    """Represents an individual acoustic panel or takeoff element with fabrication properties."""
    id: str
    tag: str
    category: str                       # e.g. 'Standard Wall Panel', 'Ceiling Cloud', 'Acoustic Baffle'
    polygon: Any                        # shapely Polygon or MultiPolygon in real-world units
    gross_area: float                   # In real-world square units (e.g. m² or sq.ft)
    cutout_area: float                  # Deducted cutout/void area
    net_area: float                     # Net billable acoustic area
    perimeter: float                    # In real-world linear units (e.g. mm, m, or ft)
    cut_width: float                    # Minimum bounding width for fabrication / nesting
    cut_length: float                   # Minimum bounding length for fabrication / nesting
    material_type: str = "Polyester 9mm"  # Material specification (Polyester 9mm, Wood Slat, etc.)
    page_number: int = 1                # Originating PDF page number (1-indexed)
    sheet_name: str = "Drawing Sheet"   # Auto-detected sheet or room name
    layer: Optional[str] = None
    stroke_color: Optional[str] = None
    fill_color: Optional[str] = None
    unit: str = "mm"
    cutout_count: int = 0
    center_x: float = 0.0
    center_y: float = 0.0
    raw_pdf_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def bounding_area(self) -> float:
        """Area of the minimum bounding rectangle used for cutting/nesting."""
        if self.unit == "mm":
            return (self.cut_width * self.cut_length) / 1_000_000.0  # m²
        elif self.unit == "m":
            return self.cut_width * self.cut_length  # m²
        elif self.unit == "inch":
            return (self.cut_width * self.cut_length) / 144.0  # sq.ft
        elif self.unit == "ft":
            return self.cut_width * self.cut_length  # sq.ft
        return self.cut_width * self.cut_length

    @property
    def dimension_str(self) -> str:
        """Formatted dimensions string e.g. '2400 × 600 mm'."""
        return f"{int(round(self.cut_length))} × {int(round(self.cut_width))} {self.unit}"


def clean_polygon(poly: Any) -> Optional[Union[Polygon, MultiPolygon]]:
    """Clean and fix self-intersecting or invalid polygons."""
    if poly is None or poly.is_empty:
        return None
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
        except Exception:
            poly = poly.buffer(0)

    if isinstance(poly, GeometryCollection):
        polys = [g for g in poly.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        poly = unary_union(polys)

    if poly.is_empty or poly.area <= 1e-6:
        return None
    return poly


def get_oriented_bounding_box(poly: Any) -> Tuple[float, float]:
    """
    Compute minimal oriented bounding box (OBB) dimensions (Length, Width)
    where Length >= Width.
    """
    if poly is None or poly.is_empty:
        return 0.0, 0.0
    try:
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        if len(coords) < 4:
            bounds = poly.bounds
            w = abs(bounds[2] - bounds[0])
            h = abs(bounds[3] - bounds[1])
            return max(w, h), min(w, h)

        side_1 = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
        side_2 = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
        return max(side_1, side_2), min(side_1, side_2)
    except Exception:
        bounds = poly.bounds
        w = abs(bounds[2] - bounds[0])
        h = abs(bounds[3] - bounds[1])
        return max(w, h), min(w, h)


def matches_any_color(candidate_hex: Optional[str], target_hexes: Optional[List[str]], tolerance: float = 0.10) -> bool:
    """Check if candidate hex matches any target hex within tolerance."""
    if not candidate_hex or not target_hexes:
        return False
    for target in target_hexes:
        if color_distance(candidate_hex, target) <= tolerance:
            return True
    return False


def vector_path_to_polygon(
    path: VectorPath,
    scale_factor: float
) -> Optional[Polygon]:
    """
    Convert a VectorPath from PDF points into a scaled Shapely Polygon.
    """
    if len(path.points) < 3:
        return None

    scaled_pts = [(pt[0] * scale_factor, pt[1] * scale_factor) for pt in path.points]

    try:
        if path.command_type == "rect":
            poly = Polygon(scaled_pts)
        elif path.is_closed or math.isclose(scaled_pts[0][0], scaled_pts[-1][0], abs_tol=1e-2):
            poly = Polygon(scaled_pts)
        else:
            p_first, p_last = scaled_pts[0], scaled_pts[-1]
            dist = math.hypot(p_first[0] - p_last[0], p_first[1] - p_last[1])
            if dist <= 5.0 * scale_factor:
                scaled_pts.append(p_first)
                poly = Polygon(scaled_pts)
            else:
                return None

        poly = clean_polygon(poly)
        return poly
    except Exception:
        return None


def reconstruct_polygons_from_lines(
    paths: List[VectorPath],
    scale_factor: float,
    snap_tolerance: float = 1.0
) -> List[Polygon]:
    """
    Reconstruct closed polygons from separate line segments using shapely polygonize.
    """
    lines = []
    for p in paths:
        if len(p.points) >= 2:
            scaled_pts = [(pt[0] * scale_factor, pt[1] * scale_factor) for pt in p.points]
            lines.append(LineString(scaled_pts))

    if not lines:
        return []

    try:
        merged = linemerge(lines)
        if snap_tolerance > 0:
            merged = snap(merged, merged, snap_tolerance)
        polys = list(polygonize(merged))
        return [clean_polygon(p) for p in polys if clean_polygon(p) is not None]
    except Exception:
        return []


def _process_page_geometry(
    page_num: int,
    sheet_name: str,
    materials_dict: Dict[str, List[VectorPath]],
    cutout_paths: List[VectorPath],
    scale_factor: float,
    to_m2: float,
    min_area_m2: float,
    max_area_m2: float,
    unit: str,
    panel_colors: Optional[List[str]] = None,
    cutout_colors: Optional[List[str]] = None,
    start_counter: int = 1
) -> Tuple[List[PanelItem], List[Polygon], int]:
    """
    Internal processor for a single page:
    - Deducts cutouts belonging to this page.
    - Classifies and computes net area and dimensions for each panel.
    """
    # 1. Convert cutouts to polygons
    page_cutout_polys: List[Polygon] = []
    for cp in cutout_paths:
        if cutout_colors:
            if not (matches_any_color(cp.stroke_hex, cutout_colors) or matches_any_color(cp.fill_hex, cutout_colors)):
                continue
        c_poly = vector_path_to_polygon(cp, scale_factor)
        if c_poly and not c_poly.is_empty:
            page_cutout_polys.append(c_poly)

    merged_cutouts = unary_union(page_cutout_polys) if page_cutout_polys else None

    page_items: List[PanelItem] = []
    current_counter = start_counter

    # 2. Iterate through each material type on this page
    for material_type, paths in materials_dict.items():
        candidates: List[Tuple[VectorPath, Polygon]] = []

        for path in paths:
            if panel_colors:
                if not (matches_any_color(path.stroke_hex, panel_colors) or matches_any_color(path.fill_hex, panel_colors)):
                    continue

            poly = vector_path_to_polygon(path, scale_factor)
            if poly is None or poly.is_empty:
                continue

            poly_area_m2 = poly.area * to_m2
            if min_area_m2 <= poly_area_m2 <= max_area_m2:
                candidates.append((path, poly))

        # Line-reconstruction fallback
        if len(candidates) == 0:
            unclosed_paths = [p for p in paths if not p.is_closed]
            if unclosed_paths:
                reconstructed = reconstruct_polygons_from_lines(unclosed_paths, scale_factor)
                for poly in reconstructed:
                    poly_area_m2 = poly.area * to_m2
                    if min_area_m2 <= poly_area_m2 <= max_area_m2:
                        dummy_p = VectorPath(
                            id=9999,
                            command_type="reconstructed",
                            points=[],
                            is_closed=True,
                            stroke_color=(0.2, 0.4, 0.8),
                            fill_color=None,
                            stroke_width=1.0,
                            bbox=(0, 0, 0, 0),
                            classified_material=material_type
                        )
                        candidates.append((dummy_p, poly))

        # Deduplicate overlapping entities
        unique_candidates: List[Tuple[VectorPath, Polygon]] = []
        for path, poly in candidates:
            dup = False
            for _, ex_poly in unique_candidates:
                if poly.equals_exact(ex_poly, tolerance=1.0) or \
                   (poly.intersection(ex_poly).area / max(poly.area, ex_poly.area) > 0.98):
                    dup = True
                    break
            if not dup:
                unique_candidates.append((path, poly))

        # Deduct cutouts and build PanelItems
        for path, poly in unique_candidates:
            gross_area_m2 = poly.area * to_m2
            net_poly = poly
            cut_area_m2 = 0.0
            n_cuts = 0

            if merged_cutouts is not None and not merged_cutouts.is_empty:
                if poly.intersects(merged_cutouts):
                    try:
                        diff = poly.difference(merged_cutouts)
                        clean_diff = clean_polygon(diff)
                        if clean_diff and not clean_diff.is_empty:
                            net_poly = clean_diff
                            cut_area_m2 = max(0.0, (poly.area - net_poly.area) * to_m2)
                            n_cuts = 1
                    except Exception:
                        pass

            net_area_m2 = net_poly.area * to_m2
            perimeter = net_poly.length
            cut_len, cut_wid = get_oriented_bounding_box(net_poly)

            # Categorize component
            aspect_ratio = cut_len / max(1.0, cut_wid)
            if aspect_ratio >= 3.5:
                category = "Acoustic Baffle / Slat"
            elif cut_len >= 2000 or cut_wid >= 1000:
                category = "Ceiling Cloud / Large Panel"
            else:
                category = "Standard Wall Panel"

            tag = f"AP-{current_counter:03d}"
            centroid = net_poly.centroid

            item = PanelItem(
                id=f"P{page_num}-{material_type[:4].upper()}-{current_counter}",
                tag=tag,
                category=category,
                material_type=material_type,
                page_number=page_num,
                sheet_name=sheet_name,
                layer=path.layer or "Default",
                stroke_color=path.stroke_hex,
                fill_color=path.fill_hex,
                polygon=net_poly,
                gross_area=round(gross_area_m2, 4),
                cutout_area=round(cut_area_m2, 4),
                net_area=round(net_area_m2, 4),
                perimeter=round(perimeter, 2),
                cut_length=round(cut_len, 1),
                cut_width=round(cut_wid, 1),
                unit=unit,
                cutout_count=n_cuts,
                center_x=centroid.x,
                center_y=centroid.y,
                raw_pdf_bbox=path.bbox
            )
            page_items.append(item)
            current_counter += 1

    return page_items, page_cutout_polys, current_counter


def process_geometry_takeoff(
    paths_or_data: Optional[Union[Dict[str, Any], List[VectorPath]]] = None,
    scale_factor: float = 17.6389,
    panel_colors: Optional[List[str]] = None,
    cutout_colors: Optional[List[str]] = None,
    min_area_m2: float = 0.05,
    max_area_m2: float = 50.0,
    unit: str = "mm",
    paths: Optional[List[VectorPath]] = None,
    document_scan_data: Optional[Dict[str, Any]] = None
) -> Tuple[List[PanelItem], List[Polygon]]:
    """
    Global Multi-Page Geometry Takeoff Engine:
    - Accepts nested page data from `scan_full_document` OR flat `List[VectorPath]`.
    - Iterates page-by-page, deducting voids and cutouts local to each page.
    - Handles pages that have NO acoustic materials gracefully (skips and moves to next page).
    - Aggregates Net Areas and returns global itemized panels grouped by material type.
    """
    # 1. Determine input structure
    scan_data = document_scan_data
    if scan_data is None and isinstance(paths_or_data, dict):
        scan_data = paths_or_data

    # Area unit converter
    if unit == "mm":
        to_m2 = 1.0 / 1_000_000.0
    elif unit == "m":
        to_m2 = 1.0
    elif unit == "inch":
        to_m2 = 0.00064516
    elif unit == "ft":
        to_m2 = 0.092903
    else:
        to_m2 = 1.0 / 1_000_000.0

    all_takeoff_items: List[PanelItem] = []
    all_cutout_polys: List[Polygon] = []
    global_counter = 1

    # Case A: Nested Multi-Page Scan Data
    if scan_data is not None and "pages" in scan_data:
        pages_dict = scan_data["pages"]

        for page_num in sorted(pages_dict.keys()):
            p_info = pages_dict[page_num]
            sheet_name = p_info.get("sheet_name", f"Sheet {page_num}")
            materials_dict = p_info.get("materials", {})
            cutout_paths = p_info.get("cutout_paths", [])

            # Gracefully skip pages with NO acoustic materials!
            total_mats = sum(len(plist) for plist in materials_dict.values())
            if total_mats == 0:
                continue

            page_items, page_cutouts, next_counter = _process_page_geometry(
                page_num=page_num,
                sheet_name=sheet_name,
                materials_dict=materials_dict,
                cutout_paths=cutout_paths,
                scale_factor=scale_factor,
                to_m2=to_m2,
                min_area_m2=min_area_m2,
                max_area_m2=max_area_m2,
                unit=unit,
                panel_colors=panel_colors,
                cutout_colors=cutout_colors,
                start_counter=global_counter
            )

            all_takeoff_items.extend(page_items)
            all_cutout_polys.extend(page_cutouts)
            global_counter = next_counter

        return all_takeoff_items, all_cutout_polys

    # Case B: Flat list of VectorPaths (Single Page / Backwards Compatibility)
    flat_paths = paths if paths is not None else (paths_or_data if isinstance(paths_or_data, list) else [])
    if not flat_paths:
        return [], []

    # Group flat paths by material
    materials_dict: Dict[str, List[VectorPath]] = {}
    cutout_paths: List[VectorPath] = []

    for p in flat_paths:
        mat_type, is_cutout = classify_vector_material(p.stroke_hex, p.fill_hex)
        if is_cutout:
            cutout_paths.append(p)
        else:
            effective_mat = mat_type or "Polyester 9mm"
            materials_dict.setdefault(effective_mat, []).append(p)

    page_items, page_cutouts, _ = _process_page_geometry(
        page_num=1,
        sheet_name="Single Page",
        materials_dict=materials_dict,
        cutout_paths=cutout_paths,
        scale_factor=scale_factor,
        to_m2=to_m2,
        min_area_m2=min_area_m2,
        max_area_m2=max_area_m2,
        unit=unit,
        panel_colors=panel_colors,
        cutout_colors=cutout_colors,
        start_counter=1
    )

    return page_items, page_cutouts


def subdivide_large_surface(
    polygon: Polygon,
    panel_w: float = 1200.0,
    panel_l: float = 2400.0,
    joint_gap: float = 5.0
) -> List[Polygon]:
    """
    Subdivide an architectural wall surface into standard acoustic panel modules.
    Useful when contractor draws an entire room wall and requires automated panel layout.
    """
    bounds = polygon.bounds
    min_x, min_y, max_x, max_y = bounds
    step_x = panel_w + joint_gap
    step_y = panel_l + joint_gap

    tiles = []
    curr_x = min_x
    while curr_x < max_x:
        curr_y = min_y
        while curr_y < max_y:
            tile_box = box(curr_x, curr_y, curr_x + panel_w, curr_y + panel_l)
            inter = polygon.intersection(tile_box)
            if not inter.is_empty and inter.area > (panel_w * panel_l * 0.05):
                cleaned = clean_polygon(inter)
                if cleaned:
                    tiles.append(cleaned)
            curr_y += step_y
        curr_x += step_x

    return tiles
