"""
Module 7: Native CAD Parser (1:1 Real-World Model Space)
Parses AutoCAD DXF/DWG files directly using ezdxf, eliminating PDF viewport scale distortion,
"dim láo" vector approximations, and rasterization errors.

Key Features:
1. True 1:1 spatial coordinates (model space millimeters).
2. Layer-wise separation of AP Polyester vs Wood Slat according to Chief QS Rules.
3. Native AutoCAD DIMENSION entity parsing (actual measurement vs text override).
4. Block (INSERT) and Cutout detection (TV screens, Door voids, Glazing openings).
5. Automatic sample DXF generator for zero-setup end-to-end testing.
6. Seamless bridge to PanelItem objects for downstream Nesting and Master BOQ.
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

try:
    import ezdxf
    from ezdxf.document import Drawing
except ImportError:
    ezdxf = None
    Drawing = None

try:
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
except ImportError:
    Polygon = None
    box = None
    unary_union = None

import importlib
try:
    _mod_geom = importlib.import_module("modules.2_geometry")
    PanelItem = _mod_geom.PanelItem
except Exception:
    PanelItem = None


# ============================================================================
# DATA SCHEMAS
# ============================================================================

class CADDimension(BaseModel):
    """Represents a native AutoCAD dimension entity."""
    dim_type: str = "Linear"
    text: str
    measurement_mm: float
    defpoint1: Tuple[float, float] = (0.0, 0.0)
    defpoint2: Tuple[float, float] = (0.0, 0.0)
    layer: str = "DIMENSIONS"
    is_override: bool = False


class CADTakeoffSummary(BaseModel):
    """Project-level summary from native CAD parsing."""
    filename: str
    dxf_version: str = "R2010"
    units: str = "mm"
    layers_found: List[str] = Field(default_factory=list)
    blocks_found: List[str] = Field(default_factory=list)
    dimensions_detected: List[CADDimension] = Field(default_factory=list)
    total_net_area_sqm: float = 0.0
    total_gross_area_sqm: float = 0.0
    total_cutout_area_sqm: float = 0.0
    total_panels_count: int = 0
    excluded_wood_slat_count: int = 0
    legend_status: Dict[str, str] = Field(default_factory=dict)


# ============================================================================
# CAD PARSER ENGINE
# ============================================================================

def is_polyester_cad_layer(layer_name: str) -> bool:
    """Classifies layer as AP Polyester according to Senior Chief QS rule."""
    norm = layer_name.upper().replace("_", " ").replace("-", " ")
    # Reject wood slat immediately
    if any(k in norm for k in ["WOOD", "SLAT", "NAN GO", "GO DUC LO", "TP603", "AC2", "AC 2", "TIMBER"]):
        return False
    # Retain polyester
    pattern = r"(?i)(\bAP\b|\bTP605\b|\bTP606\b|POLYESTER|TIEU\s*AM|ACOUSTIC\s*PANEL)"
    return bool(re.search(pattern, norm))


def is_wood_slat_cad_layer(layer_name: str) -> bool:
    """Identifies Wood Slat layers to explicitly reject them."""
    norm = layer_name.upper().replace("_", " ").replace("-", " ")
    pattern = r"(?i)(\bWOOD\b|\bSLAT\b|NAN\s*GO|GO\s*DUC\s*LO|\bTP603\b|\bAC2\b|TIMBER)"
    return bool(re.search(pattern, norm))


def is_cutout_cad_layer(layer_name: str) -> bool:
    """Identifies cutout/void layers (Doors, TV screens, Glazing)."""
    norm = layer_name.upper().replace("_", " ").replace("-", " ")
    pattern = r"(?i)(\bCUTOUT\b|\bVOID\b|\bDOOR\b|\bCUA\b|\bTV\b|SCREEN|MONITOR|OPENING|GLAZING|HVAC)"
    return bool(re.search(pattern, norm))


def parse_cad_dxf(
    dxf_source: Union[str, Path, bytes],
    project_name: str = "CAD Project Takeoff",
    target_unit: str = "mm"
) -> Tuple[CADTakeoffSummary, List[Any]]:
    """
    Directly inspects an AutoCAD DXF drawing in native 1:1 real-world space.
    Extracts closed boundaries, dimensions, and blocks; classifies layers;
    deducts cutouts; and produces standard PanelItem objects.
    """
    if ezdxf is None:
        raise ImportError("ezdxf is required for native CAD parsing. Please install ezdxf.")

    if isinstance(dxf_source, bytes):
        doc = ezdxf.read(io.StringIO(dxf_source.decode("utf-8", errors="ignore")))
        filename = "Uploaded_Drawing.dxf"
    else:
        path = Path(dxf_source)
        filename = path.name
        doc = ezdxf.readfile(str(path))

    msp = doc.modelspace()
    layers_set = {layer.dxf.name for layer in doc.layers}
    blocks_set = {b.name for b in doc.blocks if not b.name.startswith("*")}

    dimensions_found: List[CADDimension] = []
    polyester_panels: List[Any] = []
    cutout_polys: List[Tuple[str, Any, float, float]] = []  # (layer, poly, width, height)

    legend_status = {}
    for lyr in sorted(list(layers_set)):
        if is_polyester_cad_layer(lyr):
            legend_status[lyr] = "Tấm Tiêu Âm Polyester (BÓC TÁCH CHÍNH)"
        elif is_wood_slat_cad_layer(lyr):
            legend_status[lyr] = "Vách Nan Gỗ Wood Slat (LOẠI BỎ THEO QUY CHUẨN QS)"
        elif is_cutout_cad_layer(lyr):
            legend_status[lyr] = "Lỗ khoét / Mở cửa / TV (TRỪ DIỆN TÍCH)"

    # 1. Parse DIMENSION Entities
    for dim in msp.query("DIMENSION"):
        try:
            raw_text = dim.dxf.text if dim.dxf.hasattr("text") else ""
            meas = float(dim.dxf.actual_measurement) if dim.dxf.hasattr("actual_measurement") else 0.0

            # If user had text override (dim láo check)
            disp_text = raw_text.strip() if raw_text.strip() else f"{int(round(meas))}"
            is_ovr = bool(raw_text.strip() and raw_text.strip() != "<>")

            # Extract numeric value
            val_match = re.search(r"(\d+)", disp_text)
            final_val = float(val_match.group(1)) if val_match else meas

            p1 = (dim.dxf.defpoint.x, dim.dxf.defpoint.y) if dim.dxf.hasattr("defpoint") else (0.0, 0.0)
            p2 = (dim.dxf.defpoint2.x, dim.dxf.defpoint2.y) if dim.dxf.hasattr("defpoint2") else (0.0, 0.0)

            dimensions_found.append(CADDimension(
                text=disp_text,
                measurement_mm=round(final_val, 1),
                defpoint1=p1,
                defpoint2=p2,
                layer=dim.dxf.layer,
                is_override=is_ovr
            ))
        except Exception:
            continue

    # 2. Extract Cutouts (TVs, Doors, Windows)
    for ent in msp.query("LWPOLYLINE POLYLINE INSERT"):
        layer = ent.dxf.layer
        if is_cutout_cad_layer(layer):
            if ent.dxftype() in ("LWPOLYLINE", "POLYLINE"):
                pts = [(v[0], v[1]) for v in ent.get_points()]
                if len(pts) >= 3 and Polygon is not None:
                    try:
                        p_obj = Polygon(pts)
                        if p_obj.is_valid and p_obj.area > 1000.0:  # > 0.001 m²
                            minx, miny, maxx, maxy = p_obj.bounds
                            cutout_polys.append((layer, p_obj, maxx - minx, maxy - miny))
                    except Exception:
                        pass
            elif ent.dxftype() == "INSERT":
                # Block insert (e.g. TV screen block or Door void)
                ins_x = ent.dxf.insert.x
                ins_y = ent.dxf.insert.y
                # Default TV block or door size if bounding not available
                bw = 1450.0 if "TV" in layer.upper() or "TV" in ent.dxf.name.upper() else 900.0
                bh = 830.0 if "TV" in layer.upper() or "TV" in ent.dxf.name.upper() else 2100.0
                if box is not None:
                    p_obj = box(ins_x, ins_y, ins_x + bw, ins_y + bh)
                    cutout_polys.append((layer, p_obj, bw, bh))

    # 3. Extract Polyester Acoustic Panels (1:1 spatial boundary)
    panel_idx = 1
    excluded_wood_count = 0

    for ent in msp.query("LWPOLYLINE POLYLINE HATCH"):
        layer = ent.dxf.layer

        if is_wood_slat_cad_layer(layer):
            excluded_wood_count += 1
            continue

        if not is_polyester_cad_layer(layer):
            continue

        # Extract polygon vertices
        pts = []
        if ent.dxftype() in ("LWPOLYLINE", "POLYLINE"):
            pts = [(v[0], v[1]) for v in ent.get_points()]
        elif ent.dxftype() == "HATCH":
            for path in ent.paths:
                if hasattr(path, "vertices"):
                    pts = [(v[0], v[1]) for v in path.vertices]
                    break

        if len(pts) < 3 or Polygon is None:
            continue

        try:
            poly = Polygon(pts)
            if not poly.is_valid or poly.area < 10000.0:  # < 0.01 m²
                continue

            minx, miny, maxx, maxy = poly.bounds
            width_mm = round(maxx - minx, 1)
            height_mm = round(maxy - miny, 1)
            gross_m2 = round((width_mm * height_mm) / 1_000_000.0, 4)

            # Deduct overlapping cutouts
            deducted_area = 0.0
            cutout_count = 0
            curr_poly = poly

            for c_layer, c_poly, cw, ch in cutout_polys:
                if curr_poly.intersects(c_poly):
                    inter = curr_poly.intersection(c_poly)
                    if inter.area > 100.0:
                        deducted_area += (inter.area / 1_000_000.0)
                        cutout_count += 1
                        try:
                            curr_poly = curr_poly.difference(c_poly)
                        except Exception:
                            pass

            net_m2 = max(0.0, round(gross_m2 - deducted_area, 4))
            p_id = f"CAD-TP605-{panel_idx:02d}"
            tag_name = f"Vách Tiêu Âm TP605 #{panel_idx:02d}"

            if PanelItem is not None:
                p_item = PanelItem(
                    id=p_id,
                    tag=tag_name,
                    category="Acoustic Wall (CAD 1:1)",
                    polygon=curr_poly,
                    gross_area=gross_m2,
                    cutout_area=round(deducted_area, 4),
                    net_area=net_m2,
                    perimeter=round(2.0 * (width_mm + height_mm), 2),
                    cut_width=min(width_mm, height_mm),
                    cut_length=max(width_mm, height_mm),
                    material_type="TP605 (Polyester 9mm)",
                    page_number=1,
                    sheet_name=filename,
                    layer=layer,
                    stroke_color="#0284C7",
                    fill_color="#E0F2FE",
                    unit=target_unit,
                    cutout_count=cutout_count,
                    center_x=(minx + maxx) / 2.0,
                    center_y=(miny + maxy) / 2.0
                )
                polyester_panels.append(p_item)
                panel_idx += 1
        except Exception:
            continue

    tot_gross = sum(p.gross_area for p in polyester_panels)
    tot_cutout = sum(p.cutout_area for p in polyester_panels)
    tot_net = sum(p.net_area for p in polyester_panels)

    summary = CADTakeoffSummary(
        filename=filename,
        dxf_version=doc.dxfversion,
        units=target_unit,
        layers_found=sorted(list(layers_set)),
        blocks_found=sorted(list(blocks_set)),
        dimensions_detected=dimensions_found,
        total_net_area_sqm=round(tot_net, 4),
        total_gross_area_sqm=round(tot_gross, 4),
        total_cutout_area_sqm=round(tot_cutout, 4),
        total_panels_count=len(polyester_panels),
        excluded_wood_slat_count=excluded_wood_count,
        legend_status=legend_status
    )

    return summary, polyester_panels


# ============================================================================
# SAMPLE CAD DXF GENERATOR
# ============================================================================

def generate_sample_cad_dxf(output_path: Union[str, Path]) -> str:
    """
    Generates a realistic 1:1 CAD DXF file containing true-to-scale fitout entities:
    - Layer ACOUSTIC_POLYESTER_TP605: 2400 × 1200 mm Wall Panels
    - Layer CUTOUT_TV_SCREEN: 1450 × 830 mm Monitor void
    - Layer CUTOUT_DOOR_VOID: 900 × 2100 mm Exit door void
    - Layer WOOD_SLAT_TP603: Wood Slat diffuse wall (to test Chief QS exclusion)
    - Layer DIMENSIONS: Linear dimension callouts with exact 1:1 measurements.
    """
    if ezdxf is None:
        raise ImportError("ezdxf is required to generate DXF drawings.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()

    # Create CAD Layers with AEC standard colors
    doc.layers.new("ACOUSTIC_POLYESTER_TP605", dxfattribs={"color": 4})   # Cyan
    doc.layers.new("CUTOUT_TV_SCREEN", dxfattribs={"color": 1})          # Red (Cutout)
    doc.layers.new("CUTOUT_DOOR_VOID", dxfattribs={"color": 1})          # Red
    doc.layers.new("WOOD_SLAT_TP603", dxfattribs={"color": 30})          # Orange / Wood
    doc.layers.new("DIMENSIONS", dxfattribs={"color": 7})                # White/Black
    doc.layers.new("ANNOTATIONS", dxfattribs={"color": 2})               # Yellow

    # 1. Wall Panel 1 (Boardroom Meeting Wall - 2400 x 1200 mm)
    p1_pts = [(0.0, 0.0), (2400.0, 0.0), (2400.0, 1200.0), (0.0, 1200.0), (0.0, 0.0)]
    msp.add_lwpolyline(p1_pts, dxfattribs={"layer": "ACOUSTIC_POLYESTER_TP605", "closed": True})

    # Cutout 1: TV Screen inside Panel 1 (1450 x 830 mm centered)
    tv_x = (2400.0 - 1450.0) / 2.0  # 475.0
    tv_y = (1200.0 - 830.0) / 2.0   # 185.0
    tv_pts = [(tv_x, tv_y), (tv_x + 1450.0, tv_y), (tv_x + 1450.0, tv_y + 830.0), (tv_x, tv_y + 830.0), (tv_x, tv_y)]
    msp.add_lwpolyline(tv_pts, dxfattribs={"layer": "CUTOUT_TV_SCREEN", "closed": True})

    # 2. Wall Panel 2 (Studio Elevation - 2400 x 2400 mm) with Door Cutout
    p2_pts = [(2800.0, 0.0), (5200.0, 0.0), (5200.0, 2400.0), (2800.0, 2400.0), (2800.0, 0.0)]
    msp.add_lwpolyline(p2_pts, dxfattribs={"layer": "ACOUSTIC_POLYESTER_TP605", "closed": True})

    # Cutout 2: Door Opening (900 x 2100 mm)
    door_pts = [(3000.0, 0.0), (3900.0, 0.0), (3900.0, 2100.0), (3000.0, 2100.0), (3000.0, 0.0)]
    msp.add_lwpolyline(door_pts, dxfattribs={"layer": "CUTOUT_DOOR_VOID", "closed": True})

    # 3. Wood Slat Diffuser Panel (3000 x 1500 mm - Must be rejected by Chief QS Rule)
    wood_pts = [(5600.0, 0.0), (8600.0, 0.0), (8600.0, 1500.0), (5600.0, 1500.0), (5600.0, 0.0)]
    msp.add_lwpolyline(wood_pts, dxfattribs={"layer": "WOOD_SLAT_TP603", "closed": True})

    # 4. Add Linear Dimensions in 1:1 units
    msp.add_linear_dim(
        base=(1200.0, -150.0),
        p1=(0.0, 0.0),
        p2=(2400.0, 0.0),
        dimstyle="EZDXF",
        dxfattribs={"layer": "DIMENSIONS"}
    )
    msp.add_linear_dim(
        base=(-150.0, 600.0),
        p1=(0.0, 0.0),
        p2=(0.0, 1200.0),
        angle=90,
        dimstyle="EZDXF",
        dxfattribs={"layer": "DIMENSIONS"}
    )
    msp.add_linear_dim(
        base=(tv_x + 725.0, tv_y - 100.0),
        p1=(tv_x, tv_y),
        p2=(tv_x + 1450.0, tv_y),
        dimstyle="EZDXF",
        dxfattribs={"layer": "DIMENSIONS"}
    )

    # 5. Add Legend MText
    legend_text = (
        "BẢNG CHÚ THÍCH VẬT LIỆU (ACOUSTIC SPECIFICATIONS)\n"
        "Layer ACOUSTIC_POLYESTER_TP605: Tấm Tiêu Âm Sợi Polyester 9mm Xám (BÓC TÁCH)\n"
        "Layer WOOD_SLAT_TP603: Gỗ Nan Tiêu Âm Melamine (LOẠI BỎ THEO QUY CHUẨN QS)\n"
        "Layer CUTOUT_TV_SCREEN: Lỗ Khoét Tivi 1450x830 mm (TRỪ DIỆN TÍCH)\n"
        "Layer CUTOUT_DOOR_VOID: Lỗ Mở Cửa 900x2100 mm (TRỪ DIỆN TÍCH)"
    )
    msp.add_mtext(legend_text, dxfattribs={"layer": "ANNOTATIONS", "char_height": 75.0, "insert": (0.0, 2700.0)})

    doc.saveas(str(out))
    return str(out)
