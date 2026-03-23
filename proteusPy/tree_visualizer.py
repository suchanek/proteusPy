"""
TreeVisualizer: Text, PNG, and 3D rendering of DisulfideTree hierarchies.

Provides three output modes:
  1. **Text** — Unicode box-drawing for terminal / print / markdown
  2. **PNG**  — matplotlib schematic for publication / slides
  3. **3D**   — interactive pyvista scene with spheres, cylinders, and labels

All renderers walk the DisulfideTree via its ``children()`` /
``members()`` / ``node_data()`` API and require no additional
graph libraries.

Part of the program proteusPy, https://github.com/suchanek/proteusPy,
a Python package for the manipulation and analysis of macromolecules.

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

__pdoc__ = {"__all__": True}


import numpy as np

from proteusPy.disulfide_tree import DisulfideTree, TreeNodeData

# ---------------------------------------------------------------------------
# 1. Text (Unicode box-drawing) renderer
# ---------------------------------------------------------------------------

_PIPE = "\u2502"  # │
_TEE = "\u251c"  # ├
_ELBOW = "\u2514"  # └
_DASH = "\u2500"  # ─
_SPACE = " "


def _format_node_label(
    node_key: str,
    data: TreeNodeData | None,
    show_occupancy: bool = True,
    show_pct: bool = True,
) -> str:
    """Build a human-readable label for one tree node."""
    if node_key == "root":
        if data:
            return f"root ({data.occupancy:,} SS)"
        return "root"

    parts = [node_key]
    if data:
        if data.class_name:
            parts.append(data.class_name)
        if show_occupancy:
            parts.append(f"n={data.occupancy:,}")
        if show_pct:
            parts.append(f"{data.occupancy_pct:.1f}%")
    return " ".join(parts)


def _text_tree_recursive(
    tree: DisulfideTree,
    node_key: str,
    prefix: str,
    is_last: bool,
    lines: list[str],
    max_children: int,
    max_depth: int,
    current_depth: int,
    show_members: bool,
    show_occupancy: bool,
    show_pct: bool,
) -> None:
    """Recursively build text tree lines."""
    data = tree.node_data(node_key)
    label = _format_node_label(node_key, data, show_occupancy, show_pct)

    # Connector for this node
    if current_depth == 0:
        lines.append(label)
    else:
        connector = f"{_ELBOW}{_DASH}{_DASH} " if is_last else f"{_TEE}{_DASH}{_DASH} "
        lines.append(f"{prefix}{connector}{label}")

    # Update prefix for children
    if current_depth == 0:
        child_prefix = ""
    else:
        child_prefix = prefix + (_SPACE * 4 if is_last else f"{_PIPE}   ")

    if current_depth >= max_depth:
        return

    # Get children (hierarchy edges)
    children = tree.children(node_key)

    # Optionally show leaf members
    members = []
    if show_members and not children:
        members = tree.members(node_key)

    all_items = children + members
    truncated = False
    if max_children > 0 and len(all_items) > max_children:
        all_items = all_items[:max_children]
        truncated = True

    for i, child_key in enumerate(all_items):
        is_last_child = (i == len(all_items) - 1) and not truncated
        if child_key in [m for m in members]:
            # Leaf member — just show the ID
            connector = (
                f"{_ELBOW}{_DASH}{_DASH} "
                if is_last_child and not truncated
                else f"{_TEE}{_DASH}{_DASH} "
            )
            lines.append(f"{child_prefix}{connector}{child_key}")
        else:
            _text_tree_recursive(
                tree,
                child_key,
                child_prefix,
                is_last_child,
                lines,
                max_children,
                max_depth,
                current_depth + 1,
                show_members,
                show_occupancy,
                show_pct,
            )

    if truncated:
        remaining = len(children) + len(members) - max_children
        connector = f"{_ELBOW}{_DASH}{_DASH} "
        lines.append(f"{child_prefix}{connector}... +{remaining} more")


def text_tree(
    tree: DisulfideTree,
    root: str = "root",
    max_children: int = 8,
    max_depth: int = 4,
    show_members: bool = False,
    show_occupancy: bool = True,
    show_pct: bool = True,
) -> str:
    """Render a DisulfideTree as a Unicode text tree.

    Parameters
    ----------
    tree : DisulfideTree
        The tree to render.
    root : str
        Starting node key (default ``"root"``).
    max_children : int
        Maximum children to show per node (0 = unlimited).
    max_depth : int
        Maximum depth to descend (0 = root only, 4 = full hierarchy).
    show_members : bool
        Whether to show individual disulfide members at leaf nodes.
    show_occupancy : bool
        Show member counts on each node.
    show_pct : bool
        Show occupancy percentages on each node.

    Returns
    -------
    str
        Multi-line Unicode text tree.

    Examples
    --------
    >>> print(text_tree(tree, max_depth=2, max_children=3))
    root (36,456 SS)
    ├── 00200b -LHSpiral n=15,234 41.8%
    │   ├── 00200q n=8,412 23.1%
    │   ├── 00100q n=3,201 8.8%
    │   └── ... +5 more
    └── 00000b n=12,100 33.2%
    """
    lines: list[str] = []
    _text_tree_recursive(
        tree,
        root,
        "",
        True,
        lines,
        max_children,
        max_depth,
        0,
        show_members,
        show_occupancy,
        show_pct,
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. PNG (matplotlib) renderer
# ---------------------------------------------------------------------------


def _collect_layout(
    tree: DisulfideTree,
    node_key: str,
    depth: int,
    max_children: int,
    max_depth: int,
    show_members: bool,
    counter: list[int],
) -> tuple[dict, list]:
    """Walk tree and assign (x=leaf_order, y=-depth) positions.

    Returns (positions dict, edges list).
    """
    positions = {}
    edges = []

    if depth > max_depth:
        positions[node_key] = (counter[0], -depth)
        counter[0] += 1
        return positions, edges

    children = tree.children(node_key)
    members = []
    if show_members and not children:
        members = tree.members(node_key)

    all_items = children + members
    if max_children > 0 and len(all_items) > max_children:
        all_items = all_items[:max_children]

    if not all_items:
        # Leaf
        positions[node_key] = (counter[0], -depth)
        counter[0] += 1
        return positions, edges

    child_positions = {}
    for child_key in all_items:
        sub_pos, sub_edges = _collect_layout(
            tree, child_key, depth + 1, max_children, max_depth, show_members, counter
        )
        child_positions.update(sub_pos)
        positions.update(sub_pos)
        edges.extend(sub_edges)
        edges.append((node_key, child_key))

    # Parent x = mean of children x
    child_xs = [child_positions[ck][0] for ck in all_items if ck in child_positions]
    if child_xs:
        parent_x = (min(child_xs) + max(child_xs)) / 2.0
    else:
        parent_x = counter[0]
        counter[0] += 1

    positions[node_key] = (parent_x, -depth)
    return positions, edges


def _node_color(data: TreeNodeData | None, node_key: str) -> str:
    """Pick a color based on hierarchy level."""
    if node_key == "root":
        return "#2c3e50"
    if data is None:
        return "#95a5a6"  # leaf member
    level_colors = {
        "binary": "#e74c3c",
        "quadrant": "#e67e22",
        "sextant": "#f1c40f",
        "octant": "#2ecc71",
    }
    return level_colors.get(data.level, "#3498db")


def _node_label_short(node_key: str, data: TreeNodeData | None) -> str:
    """Short label for PNG nodes."""
    if node_key == "root":
        return "root"
    if data is None:
        # Leaf member — truncate long PDB IDs
        parts = node_key.split("_")
        if len(parts) >= 3:
            return f"{parts[0]}\n{parts[1]}-{parts[2]}"
        return node_key
    label = node_key
    if data.class_name:
        label += f"\n{data.class_name}"
    return label


def png_tree(
    tree: DisulfideTree,
    filename: str = "disulfide_tree.png",
    root: str = "root",
    max_children: int = 6,
    max_depth: int = 3,
    show_members: bool = False,
    figsize: tuple[float, float] = (16, 10),
    dpi: int = 150,
    title: str = "Disulfide Classification Tree",
    node_size: float = 1200,
    font_size: int = 7,
) -> str:
    """Render a DisulfideTree as a PNG schematic image.

    Uses matplotlib to draw a top-down hierarchical layout with
    color-coded nodes by level and occupancy-scaled edges.

    Parameters
    ----------
    tree : DisulfideTree
        The tree to render.
    filename : str
        Output file path (supports .png, .pdf, .svg).
    root : str
        Starting node key.
    max_children : int
        Maximum children per node.
    max_depth : int
        Maximum depth to draw.
    show_members : bool
        Whether to show leaf disulfide members.
    figsize : tuple
        Figure size in inches (width, height).
    dpi : int
        Resolution for raster output.
    title : str
        Figure title.
    node_size : float
        Base node size (matplotlib scatter units).
    font_size : int
        Label font size.

    Returns
    -------
    str
        The output filename.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    positions, edges = _collect_layout(
        tree, root, 0, max_children, max_depth, show_members, [0]
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Draw edges first (below nodes)
    for src, tgt in edges:
        if src in positions and tgt in positions:
            sx, sy = positions[src]
            tx, ty = positions[tgt]

            # Get edge weight for line thickness
            tree.node_data(src)
            tgt_data = tree.node_data(tgt)
            weight = 1.0
            if tgt_data and tgt_data.occupancy_pct > 0:
                weight = max(0.5, min(4.0, tgt_data.occupancy_pct / 5.0))

            # Orthogonal routing: vertical down, then horizontal, then vertical
            mid_y = (sy + ty) / 2.0
            ax.plot(
                [sx, sx, tx, tx],
                [sy, mid_y, mid_y, ty],
                color="#bdc3c7",
                linewidth=weight,
                solid_capstyle="round",
                zorder=1,
            )

    # Draw nodes
    for node_key, (x, y) in positions.items():
        data = tree.node_data(node_key)
        color = _node_color(data, node_key)
        label = _node_label_short(node_key, data)

        # Scale node size by occupancy
        size = node_size
        if data and data.occupancy_pct > 0:
            size = max(600, min(3000, node_size * (data.occupancy_pct / 10.0)))

        ax.scatter(x, y, s=size, c=color, zorder=3, edgecolors="white", linewidths=1.5)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=font_size,
            color="white",
            fontweight="bold",
            zorder=4,
        )

        # Occupancy annotation below node
        if data and data.occupancy > 0 and node_key != "root":
            ax.text(
                x,
                y - 0.35,
                f"n={data.occupancy:,}",
                ha="center",
                va="top",
                fontsize=font_size - 1,
                color="#7f8c8d",
                zorder=4,
            )

    # Legend
    legend_items = [
        mpatches.Patch(color="#2c3e50", label="Root"),
        mpatches.Patch(color="#e74c3c", label="Binary (2\u2075)"),
        mpatches.Patch(color="#e67e22", label="Quadrant (4\u2075)"),
        mpatches.Patch(color="#f1c40f", label="Sextant (6\u2075)"),
        mpatches.Patch(color="#2ecc71", label="Octant (8\u2075)"),
    ]
    if show_members:
        legend_items.append(mpatches.Patch(color="#95a5a6", label="Member"))

    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return filename


# ---------------------------------------------------------------------------
# 3. Interactive 3D (pyvista) renderer
# ---------------------------------------------------------------------------

#: Level colors as RGB floats for pyvista meshes.
_LEVEL_COLORS_RGB = {
    "root": (0.173, 0.243, 0.314),  # #2c3e50
    "binary": (0.906, 0.298, 0.235),  # #e74c3c
    "quadrant": (0.902, 0.494, 0.133),  # #e67e22
    "sextant": (0.945, 0.769, 0.059),  # #f1c40f
    "octant": (0.180, 0.800, 0.443),  # #2ecc71
    "member": (0.584, 0.647, 0.651),  # #95a5a6
}

#: Y spacing between hierarchy levels in the 3D scene.
_LEVEL_Y_SPACING = 3.0

#: Base sphere radius for nodes.
_BASE_SPHERE_RADIUS = 0.3

#: Cylinder radius for edges.
_EDGE_RADIUS = 0.04


def _collect_layout_3d(
    tree: DisulfideTree,
    node_key: str,
    depth: int,
    max_children: int,
    max_depth: int,
    show_members: bool,
    counter: list[float],
    x_spacing: float,
) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    """Walk tree and assign 3D positions (x=leaf_order, y=-depth, z=0).

    Returns (positions dict, edges list).
    """
    positions: dict[str, np.ndarray] = {}
    edges: list[tuple[str, str]] = []

    if depth > max_depth:
        positions[node_key] = np.array(
            [counter[0] * x_spacing, -depth * _LEVEL_Y_SPACING, 0.0]
        )
        counter[0] += 1
        return positions, edges

    children = tree.children(node_key)
    members = []
    if show_members and not children:
        members = tree.members(node_key)

    all_items = children + members
    if max_children > 0 and len(all_items) > max_children:
        all_items = all_items[:max_children]

    if not all_items:
        positions[node_key] = np.array(
            [counter[0] * x_spacing, -depth * _LEVEL_Y_SPACING, 0.0]
        )
        counter[0] += 1
        return positions, edges

    child_positions: dict[str, np.ndarray] = {}
    for child_key in all_items:
        sub_pos, sub_edges = _collect_layout_3d(
            tree,
            child_key,
            depth + 1,
            max_children,
            max_depth,
            show_members,
            counter,
            x_spacing,
        )
        child_positions.update(sub_pos)
        positions.update(sub_pos)
        edges.extend(sub_edges)
        edges.append((node_key, child_key))

    # Parent x = mean of children x
    child_xs = [child_positions[ck][0] for ck in all_items if ck in child_positions]
    if child_xs:
        parent_x = (min(child_xs) + max(child_xs)) / 2.0
    else:
        parent_x = counter[0] * x_spacing
        counter[0] += 1

    positions[node_key] = np.array(
        [parent_x, -depth * _LEVEL_Y_SPACING, 0.0]
    )
    return positions, edges


def _node_color_rgb(
    data: TreeNodeData | None, node_key: str
) -> tuple[float, float, float]:
    """Return an RGB tuple for a tree node."""
    if node_key == "root":
        return _LEVEL_COLORS_RGB["root"]
    if data is None:
        return _LEVEL_COLORS_RGB["member"]
    return _LEVEL_COLORS_RGB.get(data.level, (0.204, 0.596, 0.859))


def _node_label_3d(node_key: str, data: TreeNodeData | None) -> str:
    """Compact label for 3D point labels."""
    if node_key == "root":
        if data:
            return f"root ({data.occupancy:,})"
        return "root"
    if data is None:
        return node_key
    label = node_key
    if data.class_name:
        label += f" {data.class_name}"
    label += f" n={data.occupancy:,}"
    return label


def tree_3d(
    tree: DisulfideTree,
    root: str = "root",
    max_children: int = 6,
    max_depth: int = 3,
    show_members: bool = False,
    title: str = "Disulfide Classification Tree",
    x_spacing: float = 2.0,
    shadows: bool = False,
    light: str = "auto",
    off_screen: bool = False,
    screenshot: str | None = None,
    window_size: tuple[int, int] = (1024, 1024),
) -> str | None:
    """Render a DisulfideTree as an interactive 3D pyvista scene.

    Nodes are drawn as spheres sized by occupancy percentage, colored
    by hierarchy level.  Edges are drawn as cylinders with thickness
    proportional to child occupancy.  Labels are placed above each
    sphere.

    Parameters
    ----------
    tree : DisulfideTree
        The tree to render.
    root : str
        Starting node key.
    max_children : int
        Maximum children per node.
    max_depth : int
        Maximum depth to draw.
    show_members : bool
        Whether to show leaf disulfide members.
    title : str
        Scene title.
    x_spacing : float
        Horizontal spacing between leaf nodes.
    shadows : bool
        Enable shadow rendering.
    light : str
        PyVista theme: ``"auto"``, ``"light"``, or ``"dark"``.
    off_screen : bool
        If True, render without displaying a window (for screenshots).
    screenshot : str, optional
        If given, save a screenshot to this path and close. Implies
        ``off_screen=True``.
    window_size : tuple[int, int]
        Window size in pixels.

    Returns
    -------
    str or None
        The screenshot filename if saved, else None.
    """
    import pyvista as pv

    from proteusPy.utility import dpi_adjusted_fontsize, set_pyvista_theme

    set_pyvista_theme(light)

    if screenshot:
        off_screen = True

    # Collect positions and edges
    positions, edges = _collect_layout_3d(
        tree, root, 0, max_children, max_depth, show_members, [0], x_spacing
    )

    pl = pv.Plotter(window_size=window_size, off_screen=off_screen)
    pl.add_title(title=title, font_size=dpi_adjusted_fontsize(10))
    pl.enable_anti_aliasing("msaa")

    # --- Draw edges as cylinders ---
    for src, tgt in edges:
        if src not in positions or tgt not in positions:
            continue

        src_pos = positions[src]
        tgt_pos = positions[tgt]

        direction = tgt_pos - src_pos
        height = float(np.linalg.norm(direction))
        if height < 1e-6:
            continue

        midpoint = (src_pos + tgt_pos) / 2.0

        # Edge thickness from child occupancy
        tgt_data = tree.node_data(tgt)
        edge_radius = _EDGE_RADIUS
        if tgt_data and tgt_data.occupancy_pct > 0:
            edge_radius = max(0.02, min(0.12, _EDGE_RADIUS * tgt_data.occupancy_pct / 3.0))

        cyl = pv.Cylinder(
            center=midpoint.tolist(),
            direction=direction.tolist(),
            radius=edge_radius,
            height=height,
            capping=True,
            resolution=16,
        )
        pl.add_mesh(cyl, color="#bdc3c7", smooth_shading=True)

    # --- Draw nodes as spheres ---
    for node_key, pos in positions.items():
        data = tree.node_data(node_key)
        color = _node_color_rgb(data, node_key)
        label = _node_label_3d(node_key, data)

        # Scale radius by occupancy
        radius = _BASE_SPHERE_RADIUS
        if data and data.occupancy_pct > 0:
            radius = max(
                0.15,
                min(0.8, _BASE_SPHERE_RADIUS * (data.occupancy_pct / 5.0)),
            )

        sphere = pv.Sphere(
            center=pos.tolist(),
            radius=radius,
            theta_resolution=24,
            phi_resolution=24,
        )
        pl.add_mesh(
            sphere,
            color=color,
            smooth_shading=True,
            specular=0.7,
            specular_power=80,
        )

        # Label above sphere
        label_pos = pos.copy()
        label_pos[1] += radius + 0.2
        pl.add_point_labels(
            [label_pos.tolist()],
            [label],
            font_size=8,
            point_size=0,
            bold=True,
            text_color=color,
            shape_opacity=0.0,
            always_visible=True,
        )

    # Camera: isometric view from above-front
    pl.camera_position = "iso"
    pl.reset_camera()

    if shadows:
        pl.enable_shadows()

    if screenshot:
        try:
            pl.screenshot(screenshot)
        except RuntimeError:
            pass
        pl.close()
        return screenshot

    pl.show()
    return None


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "text_tree",
    "png_tree",
    "tree_3d",
]
