"""
Module 3: 2D Bin Packing & Global Material Nesting Optimizer
Packs acoustic panels independently for each material type across the entire project
into standard stock sheets (e.g. 2400x1200mm) to maximize material yield globally,
minimize scrap/offcut waste, and produce visual fabrication cut plans.
"""

from dataclasses import dataclass, field
import io
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import rectpack
    from rectpack import (
        PackingAlgorithm,
        PackingBin,
        MaxRectsBl,
        MaxRectsBssf,
        MaxRectsBaf,
        MaxRectsLssf,
        SkylineBl,
        SkylineMwf,
        GuillotineBssfSas,
        newPacker
    )
except ImportError:
    rectpack = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image
except ImportError:
    plt = None
    Image = None

import importlib

_mod_geom = importlib.import_module("modules.2_geometry")
PanelItem = _mod_geom.PanelItem


@dataclass
class NestedPiece:
    """Represents a panel piece placed on a stock sheet."""
    piece_id: str
    tag: str
    category: str
    x: float
    y: float
    width: float
    height: float
    is_rotated: bool
    original_length: float
    original_width: float
    net_area: float
    material_type: str = "Polyester 9mm"
    page_number: int = 1
    sheet_name: str = "Drawing Sheet"


@dataclass
class OffcutRemnant:
    """Represents a usable remnant rectangle from the unused portion of a sheet."""
    x: float
    y: float
    width: float
    height: float
    area: float
    is_usable: bool


@dataclass
class NestedSheet:
    """Represents a single stock sheet with placed pieces and waste analytics."""
    sheet_index: int
    stock_width: float
    stock_height: float
    unit: str
    material_type: str = "Polyester 9mm"
    pieces: List[NestedPiece] = field(default_factory=list)
    remnants: List[OffcutRemnant] = field(default_factory=list)

    @property
    def sheet_area(self) -> float:
        return self.stock_width * self.stock_height

    @property
    def used_area(self) -> float:
        return sum(p.width * p.height for p in self.pieces)

    @property
    def waste_area(self) -> float:
        return max(0.0, self.sheet_area - self.used_area)

    @property
    def yield_pct(self) -> float:
        if self.sheet_area <= 0:
            return 0.0
        return round((self.used_area / self.sheet_area) * 100.0, 2)

    @property
    def waste_pct(self) -> float:
        return round(100.0 - self.yield_pct, 2)


@dataclass
class NestingResult:
    """Nesting results for a specific material type or single batch."""
    material_type: str = "All Materials"
    sheets: List[NestedSheet] = field(default_factory=list)
    unplaced_items: List[PanelItem] = field(default_factory=list)
    stock_width: float = 2400.0
    stock_height: float = 1200.0
    unit: str = "mm"
    blade_kerf: float = 3.0
    rotation_allowed: bool = True
    algorithm_name: str = "MaxRectsBssf"

    @property
    def total_sheets(self) -> int:
        return len(self.sheets)

    @property
    def total_placed_pieces(self) -> int:
        return sum(len(s.pieces) for s in self.sheets)

    @property
    def total_stock_area_m2(self) -> float:
        to_m2 = 1.0 / 1_000_000.0 if self.unit == "mm" else 1.0
        return round(sum(s.sheet_area for s in self.sheets) * to_m2, 4)

    @property
    def total_used_area_m2(self) -> float:
        to_m2 = 1.0 / 1_000_000.0 if self.unit == "mm" else 1.0
        return round(sum(s.used_area for s in self.sheets) * to_m2, 4)

    @property
    def total_waste_area_m2(self) -> float:
        return round(max(0.0, self.total_stock_area_m2 - self.total_used_area_m2), 4)

    @property
    def overall_yield_pct(self) -> float:
        if self.total_stock_area_m2 <= 0:
            return 0.0
        return round((self.total_used_area_m2 / self.total_stock_area_m2) * 100.0, 2)

    @property
    def overall_waste_pct(self) -> float:
        return round(100.0 - self.overall_yield_pct, 2)


@dataclass
class GlobalNestingResult:
    """
    Project-wide nesting container holding independent nesting results
    grouped by Material Type.
    """
    material_results: Dict[str, NestingResult] = field(default_factory=dict)
    stock_width: float = 2400.0
    stock_height: float = 1200.0
    unit: str = "mm"
    blade_kerf: float = 3.0
    rotation_allowed: bool = True
    algorithm_name: str = "MaxRectsBssf"

    @property
    def sheets(self) -> List[NestedSheet]:
        """Flattened list of all sheets across all materials for backwards compatibility."""
        all_s = []
        for res in self.material_results.values():
            all_s.extend(res.sheets)
        return all_s

    @property
    def unplaced_items(self) -> List[PanelItem]:
        all_unplaced = []
        for res in self.material_results.values():
            all_unplaced.extend(res.unplaced_items)
        return all_unplaced

    @property
    def total_sheets(self) -> int:
        return sum(res.total_sheets for res in self.material_results.values())

    @property
    def total_placed_pieces(self) -> int:
        return sum(res.total_placed_pieces for res in self.material_results.values())

    @property
    def total_stock_area_m2(self) -> float:
        return round(sum(res.total_stock_area_m2 for res in self.material_results.values()), 4)

    @property
    def total_used_area_m2(self) -> float:
        return round(sum(res.total_used_area_m2 for res in self.material_results.values()), 4)

    @property
    def total_waste_area_m2(self) -> float:
        return round(max(0.0, self.total_stock_area_m2 - self.total_used_area_m2), 4)

    @property
    def overall_yield_pct(self) -> float:
        if self.total_stock_area_m2 <= 0:
            return 0.0
        return round((self.total_used_area_m2 / self.total_stock_area_m2) * 100.0, 2)

    @property
    def overall_waste_pct(self) -> float:
        return round(100.0 - self.overall_yield_pct, 2)


def calculate_remnants(
    stock_w: float,
    stock_h: float,
    pieces: List[NestedPiece],
    min_usable_dim: float = 500.0
) -> List[OffcutRemnant]:
    """Calculate bounding offcut remnants from remaining sheet space."""
    if not pieces:
        return [OffcutRemnant(0, 0, stock_w, stock_h, stock_w * stock_h, True)]

    max_x = max(p.x + p.width for p in pieces)
    max_y = max(p.y + p.height for p in pieces)
    remnants = []

    # Right remnant
    if stock_w - max_x >= 50:
        rem_w = stock_w - max_x
        rem_h = stock_h
        area = rem_w * rem_h
        remnants.append(
            OffcutRemnant(
                x=round(max_x, 1),
                y=0.0,
                width=round(rem_w, 1),
                height=round(rem_h, 1),
                area=round(area, 1),
                is_usable=(rem_w >= min_usable_dim and rem_h >= min_usable_dim)
            )
        )

    # Top remnant
    if stock_h - max_y >= 50:
        rem_w = max_x
        rem_h = stock_h - max_y
        area = rem_w * rem_h
        remnants.append(
            OffcutRemnant(
                x=0.0,
                y=round(max_y, 1),
                width=round(rem_w, 1),
                height=round(rem_h, 1),
                area=round(area, 1),
                is_usable=(rem_w >= min_usable_dim and rem_h >= min_usable_dim)
            )
        )

    return remnants


def _pack_single_material(
    panels: List[PanelItem],
    material_name: str,
    stock_width: float,
    stock_height: float,
    unit: str,
    blade_kerf: float,
    allow_rotation: bool,
    algorithm_name: str,
    sheet_margin: float,
    max_sheets: int
) -> NestingResult:
    """Run 2D bin packing on a single material type."""
    if not panels:
        return NestingResult(
            material_type=material_name,
            stock_width=stock_width,
            stock_height=stock_height,
            unit=unit
        )

    usable_w = stock_width - (2 * sheet_margin)
    usable_h = stock_height - (2 * sheet_margin)

    algo_map = {}
    if rectpack is not None:
        algo_map = {
            "MaxRectsBssf": MaxRectsBssf,
            "MaxRectsBl": MaxRectsBl,
            "MaxRectsBaf": MaxRectsBaf,
            "MaxRectsLssf": MaxRectsLssf,
            "SkylineBl": SkylineBl,
            "SkylineMwf": SkylineMwf,
            "GuillotineBssfSas": GuillotineBssfSas
        }
    pack_algo = algo_map.get(algorithm_name, MaxRectsBssf if rectpack else None)

    if rectpack is not None and pack_algo is not None:
        packer = newPacker(
            mode=rectpack.PackingMode.Bbox,
            pack_algo=pack_algo,
            rotation=allow_rotation
        )

        to_m2 = 1e6 if unit == "mm" else 1.0
        est_sheets = max(5, math.ceil(sum(p.bounding_area for p in panels) / ((usable_w * usable_h) / to_m2) * 1.5))
        for b_idx in range(min(max_sheets, est_sheets + 15)):
            packer.add_bin(usable_w, usable_h, count=1, bid=b_idx)

        item_lookup: Dict[str, PanelItem] = {}
        for p in panels:
            item_lookup[p.id] = p
            w_kerf = p.cut_length + blade_kerf
            h_kerf = p.cut_width + blade_kerf
            packer.add_rect(w_kerf, h_kerf, rid=p.id)

        packer.pack()

        sheets_dict: Dict[int, List[NestedPiece]] = {}
        placed_ids = set()

        for abin in packer:
            b_id = abin.bid
            for rect in abin:
                rx, ry, rw, rh, rid = rect.x, rect.y, rect.width, rect.height, rect.rid
                placed_ids.add(rid)
                orig_item = item_lookup[rid]

                pw = rw - blade_kerf
                ph = rh - blade_kerf

                is_rot = False
                if math.isclose(pw, orig_item.cut_width, abs_tol=blade_kerf + 1.0) and \
                   math.isclose(ph, orig_item.cut_length, abs_tol=blade_kerf + 1.0):
                    is_rot = True

                npiece = NestedPiece(
                    piece_id=orig_item.id,
                    tag=orig_item.tag,
                    category=orig_item.category,
                    x=rx + sheet_margin,
                    y=ry + sheet_margin,
                    width=pw,
                    height=ph,
                    is_rotated=is_rot,
                    original_length=orig_item.cut_length,
                    original_width=orig_item.cut_width,
                    net_area=orig_item.net_area,
                    material_type=orig_item.material_type,
                    page_number=orig_item.page_number,
                    sheet_name=orig_item.sheet_name
                )

                sheets_dict.setdefault(b_id, []).append(npiece)

        unplaced = [p for p in panels if p.id not in placed_ids]

        nested_sheets = []
        for s_idx, (b_id, p_list) in enumerate(sorted(sheets_dict.items()), start=1):
            remnants = calculate_remnants(stock_width, stock_height, p_list)
            ns = NestedSheet(
                sheet_index=s_idx,
                stock_width=stock_width,
                stock_height=stock_height,
                unit=unit,
                material_type=material_name,
                pieces=p_list,
                remnants=remnants
            )
            nested_sheets.append(ns)

        return NestingResult(
            material_type=material_name,
            sheets=nested_sheets,
            unplaced_items=unplaced,
            stock_width=stock_width,
            stock_height=stock_height,
            unit=unit,
            blade_kerf=blade_kerf,
            rotation_allowed=allow_rotation,
            algorithm_name=algorithm_name
        )

    else:
        # Fallback Greedy Shelf-First-Fit Packer
        return _run_heuristic_shelf_pack(
            panels=panels,
            material_name=material_name,
            stock_width=stock_width,
            stock_height=stock_height,
            unit=unit,
            blade_kerf=blade_kerf,
            sheet_margin=sheet_margin,
            allow_rotation=allow_rotation
        )


def _run_heuristic_shelf_pack(
    panels: List[PanelItem],
    material_name: str,
    stock_width: float,
    stock_height: float,
    unit: str,
    blade_kerf: float,
    sheet_margin: float,
    allow_rotation: bool
) -> NestingResult:
    """Heuristic Shelf Packing fallback algorithm."""
    usable_w = stock_width - (2 * sheet_margin)
    usable_h = stock_height - (2 * sheet_margin)

    sorted_panels = sorted(panels, key=lambda p: max(p.cut_length, p.cut_width), reverse=True)

    nested_sheets: List[NestedSheet] = []
    unplaced: List[PanelItem] = []

    current_sheet_idx = 1
    current_pieces: List[NestedPiece] = []
    curr_x = 0.0
    curr_y = 0.0
    shelf_height = 0.0

    for p in sorted_panels:
        pw = p.cut_length
        ph = p.cut_width
        is_rot = False

        if allow_rotation and pw > usable_w and ph <= usable_w:
            pw, ph = ph, pw
            is_rot = True

        if pw + blade_kerf > usable_w or ph + blade_kerf > usable_h:
            unplaced.append(p)
            continue

        if curr_x + pw + blade_kerf <= usable_w and curr_y + ph + blade_kerf <= usable_h:
            np = NestedPiece(
                piece_id=p.id,
                tag=p.tag,
                category=p.category,
                x=curr_x + sheet_margin,
                y=curr_y + sheet_margin,
                width=pw,
                height=ph,
                is_rotated=is_rot,
                original_length=p.cut_length,
                original_width=p.cut_width,
                net_area=p.net_area,
                material_type=p.material_type,
                page_number=p.page_number,
                sheet_name=p.sheet_name
            )
            current_pieces.append(np)
            curr_x += pw + blade_kerf
            shelf_height = max(shelf_height, ph + blade_kerf)
        else:
            curr_x = 0.0
            curr_y += shelf_height
            shelf_height = 0.0

            if curr_y + ph + blade_kerf <= usable_h:
                np = NestedPiece(
                    piece_id=p.id,
                    tag=p.tag,
                    category=p.category,
                    x=curr_x + sheet_margin,
                    y=curr_y + sheet_margin,
                    width=pw,
                    height=ph,
                    is_rotated=is_rot,
                    original_length=p.cut_length,
                    original_width=p.cut_width,
                    net_area=p.net_area,
                    material_type=p.material_type,
                    page_number=p.page_number,
                    sheet_name=p.sheet_name
                )
                current_pieces.append(np)
                curr_x += pw + blade_kerf
                shelf_height = ph + blade_kerf
            else:
                remnants = calculate_remnants(stock_width, stock_height, current_pieces)
                nested_sheets.append(
                    NestedSheet(
                        sheet_index=current_sheet_idx,
                        stock_width=stock_width,
                        stock_height=stock_height,
                        unit=unit,
                        material_type=material_name,
                        pieces=current_pieces,
                        remnants=remnants
                    )
                )
                current_sheet_idx += 1
                current_pieces = []
                curr_x = 0.0
                curr_y = 0.0
                shelf_height = 0.0

                np = NestedPiece(
                    piece_id=p.id,
                    tag=p.tag,
                    category=p.category,
                    x=curr_x + sheet_margin,
                    y=curr_y + sheet_margin,
                    width=pw,
                    height=ph,
                    is_rotated=is_rot,
                    original_length=p.cut_length,
                    original_width=p.cut_width,
                    net_area=p.net_area,
                    material_type=p.material_type,
                    page_number=p.page_number,
                    sheet_name=p.sheet_name
                )
                current_pieces.append(np)
                curr_x += pw + blade_kerf
                shelf_height = ph + blade_kerf

    if current_pieces:
        remnants = calculate_remnants(stock_width, stock_height, current_pieces)
        nested_sheets.append(
            NestedSheet(
                sheet_index=current_sheet_idx,
                stock_width=stock_width,
                stock_height=stock_height,
                unit=unit,
                material_type=material_name,
                pieces=current_pieces,
                remnants=remnants
            )
        )

    return NestingResult(
        material_type=material_name,
        sheets=nested_sheets,
        unplaced_items=unplaced,
        stock_width=stock_width,
        stock_height=stock_height,
        unit=unit,
        blade_kerf=blade_kerf,
        rotation_allowed=allow_rotation,
        algorithm_name="HeuristicShelfPack"
    )


def run_nesting_optimization(
    panels: List[PanelItem],
    stock_width: float = 2400.0,
    stock_height: float = 1200.0,
    unit: str = "mm",
    blade_kerf: float = 3.0,
    allow_rotation: bool = True,
    algorithm_name: str = "MaxRectsBssf",
    sheet_margin: float = 10.0,
    max_sheets: int = 500
) -> GlobalNestingResult:
    """
    Refactored Global Nesting Engine:
    - Runs 2D bin packing INDEPENDENTLY for each material type across the whole project.
      (e.g., Nests all 'Polyester 9mm' panels from Page 1, 3, and 5 together to optimize yield globally).
    - Returns a GlobalNestingResult holding per-material results and global KPI aggregates.
    """
    if not panels:
        return GlobalNestingResult(
            material_results={},
            stock_width=stock_width,
            stock_height=stock_height,
            unit=unit,
            blade_kerf=blade_kerf,
            rotation_allowed=allow_rotation,
            algorithm_name=algorithm_name
        )

    # 1. Group panels across all pages by material_type
    materials_dict: Dict[str, List[PanelItem]] = {}
    for p in panels:
        mat = p.material_type or "Polyester 9mm"
        materials_dict.setdefault(mat, []).append(p)

    # 2. Run packing independently for each material
    global_results: Dict[str, NestingResult] = {}
    for mat_name, p_list in sorted(materials_dict.items()):
        mat_res = _pack_single_material(
            panels=p_list,
            material_name=mat_name,
            stock_width=stock_width,
            stock_height=stock_height,
            unit=unit,
            blade_kerf=blade_kerf,
            allow_rotation=allow_rotation,
            algorithm_name=algorithm_name,
            sheet_margin=sheet_margin,
            max_sheets=max_sheets
        )
        global_results[mat_name] = mat_res

    return GlobalNestingResult(
        material_results=global_results,
        stock_width=stock_width,
        stock_height=stock_height,
        unit=unit,
        blade_kerf=blade_kerf,
        rotation_allowed=allow_rotation,
        algorithm_name=algorithm_name
    )


def render_nested_sheet_diagram(
    sheet: NestedSheet,
    dpi: int = 150
) -> Any:
    """
    Renders an engineering cut layout diagram for a single stock sheet using Matplotlib.
    Displays sheet index, material type, pieces, yield, and waste.
    Returns a PIL Image.
    """
    if plt is None or Image is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=dpi)

    sheet_rect = patches.Rectangle(
        (0, 0), sheet.stock_width, sheet.stock_height,
        linewidth=2.0, edgecolor="#1E293B", facecolor="#F8FAFC", zorder=1
    )
    ax.add_patch(sheet_rect)

    # Material specific color theme
    mat_color_map = {
        "Polyester 9mm": ("#0284C7", "#E0F2FE"),       # Blue
        "Wood Slat Panel": ("#D97706", "#FEF3C7"),     # Amber
        "Fabric Wrapped 25mm": ("#059669", "#D1FAE5"), # Emerald
    }
    stroke_c, fill_c = mat_color_map.get(sheet.material_type, ("#475569", "#F1F5F9"))

    for piece in sheet.pieces:
        p_rect = patches.Rectangle(
            (piece.x, piece.y), piece.width, piece.height,
            linewidth=1.2, edgecolor=stroke_c, facecolor=fill_c, zorder=2
        )
        ax.add_patch(p_rect)

        cx = piece.x + piece.width / 2.0
        cy = piece.y + piece.height / 2.0

        label_text = f"{piece.tag} [P.{piece.page_number}]\n{int(round(piece.width))}×{int(round(piece.height))} {sheet.unit}"
        if piece.is_rotated:
            label_text += " ⟲"

        fontsize = 7.0 if min(piece.width, piece.height) > 150 else 5.8
        ax.text(
            cx, cy, label_text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="#0F172A", zorder=3
        )

    # Draw usable remnants as shaded hatched regions
    for rem in sheet.remnants:
        if rem.is_usable:
            r_patch = patches.Rectangle(
                (rem.x, rem.y), rem.width, rem.height,
                linewidth=1.0, edgecolor="#94A3B8", facecolor="#E2E8F0",
                linestyle="--", hatch="//", alpha=0.5, zorder=1.5
            )
            ax.add_patch(r_patch)
            ax.text(
                rem.x + rem.width / 2.0, rem.y + rem.height / 2.0,
                f"REMNANT OFF-CUT\n{int(round(rem.width))}×{int(round(rem.height))} {sheet.unit}",
                ha="center", va="center", fontsize=6.5, color="#64748B", style="italic", zorder=2
            )

    ax.set_xlim(-50, sheet.stock_width + 50)
    ax.set_ylim(-50, sheet.stock_height + 50)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    title_str = (
        f"STOCK SHEET #{sheet.sheet_index} [{sheet.material_type.upper()}] — CUTTING & NESTING PATTERN\n"
        f"Stock Size: {int(sheet.stock_width)} × {int(sheet.stock_height)} {sheet.unit}  |  "
        f"Pieces: {len(sheet.pieces)}  |  "
        f"Yield: {sheet.yield_pct}%  |  Waste: {sheet.waste_pct}%"
    )
    ax.set_title(title_str, fontsize=9.5, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xlabel(f"Sheet Width ({sheet.unit})", fontsize=8)
    ax.set_ylabel(f"Sheet Height ({sheet.unit})", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
