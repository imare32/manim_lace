"""
将任意 Manim 路径转换为具有"编织交错"效果的丝带图案
"""

import functools
import time

import networkx
import numpy as np
from manim import Graph, Polygon, VGroup, VMobject
from manim.utils.color import BLUE, RED, WHITE, BLACK
from matplotlib.path import Path as MatplotlibPath
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .swatches import random_swatch

DEV_MODE = 1


def timer(func):
    """Profiling 装饰器"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if DEV_MODE:
            start_time = time.perf_counter()
            result = func(self, *args, **kwargs)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            print(f"⏱️ 耗时: {elapsed:.4f} 秒[{func.__qualname__}] ")
            return result
        else:
            return func(self, *args, **kwargs)

    return wrapper


# ============================================================
# 数据结构
# ============================================================


class Anchor:
    """两条 PolyLineSegment 的交点，或 PolyLineSegment 的端点"""

    def __init__(
        self,
        coord: NDArray[np.float64],
        seg1: "PolyLineSegment",
        seg2: "PolyLineSegment | None" = None,
    ):
        self.id = id(self)
        self.coord = coord
        self.seg1 = seg1
        self.seg2 = seg2

    @property
    def is_cap(self):
        """是否为端点"""
        return self.seg2 is None


class Link:
    """两个 Anchor 之间的线段"""

    def __init__(self, start: Anchor, end: Anchor):
        self.start, self.end = start, end
        self.z_index: int = 0
        self.is_overlap = False  # 是否在交叉区域
        self.overlap: "Overlap" = None
        self.twin: "Link" = None  # 对应的另一条偏移线上的 AnchorSegment


class Overlap:
    """Overlap：交叉区域，由四条 is_overlap=True 的 AnchorSegment 组成"""

    def __init__(self):
        self.visited = False
        self.drawable = True
        self.links: list[Link] = []
    
    @property
    def center(self):
        """返回交叉区域的中心坐标"""
        coords = [link.start.coord for link in self.links] + [link.end.coord for link in self.links]
        return np.mean(coords, axis=0)


class PolyLineSegment:
    """Polyline 的一段, 与其他 PolyLineSegment 相交"""

    def __init__(self, coord1: NDArray[np.float64], coord2: NDArray[np.float64]):
        self.id = id(self)
        self.coord1, self.coord2 = coord1, coord2
        self.anchors: list[Anchor] = []
        self.links: list[Link] = []

    def _sort_anchors(self):
        """按离`起点坐标`的距离排序交点"""
        self.anchors.sort(key=lambda anchor: np.linalg.norm(anchor.coord - self.coord1))


class Polyline:
    """折线或者多边形"""

    def __init__(self, coords: NDArray[np.float64], closed=True):
        self.id = id(self)
        self.coords = coords
        self.closed = closed
        self.segments: list[PolyLineSegment] = []
        if self.closed:
            for p1, p2 in zip(coords, np.roll(coords, -1, axis=0)):
                self.segments.append(PolyLineSegment(p1, p2))
        else:
            for p1, p2 in zip(coords, coords[1:]):
                self.segments.append(PolyLineSegment(p1, p2))
            # 为开放线段设置端点
            self.segments[0].anchors.append(Anchor(self.coords[0], self.segments[0]))
            self.segments[-1].anchors.append(Anchor(self.coords[-1], self.segments[-1]))

    def iter_links(self):
        for seg in self.segments:
            yield from seg.links


class ParallelPolyline:
    """ParallelPolyline：主线 + 两条偏移线"""

    def __init__(self, polyline: Polyline, offset: float):
        self.id = id(self)
        self.polyline = polyline
        self.offset = offset
        self._generate_offset_lines()

    def _generate_offset_lines(self):
        """生成偏移线"""
        closed = self.polyline.closed
        poly1, poly2 = generate_offset_lines(self.polyline.coords, self.offset, closed)
        self.offset_polylines = Polyline(poly1, closed), Polyline(poly2, closed)
        self.polylines = (self.polyline, *self.offset_polylines)


class HoleFragment:
    """丝带交错编织产生的孔洞,对于开放边缘,孔洞会智能缝合"""

    def __init__(self, coords: NDArray[np.float64]):
        self.coords = coords


class RibbonFragment:
    """交错编织的丝带被剪开后的多边形"""

    def __init__(self, coords: NDArray[np.float64]):
        self.coords = coords


# ============================================================
# Lace 主类
# ============================================================


class Lace(VGroup):
    @timer
    def __init__(
        self,
        target_mobject: VMobject,
        offset=0.1,
        atol=1e-6,
        hole_opacity=1.0,
        hole_color=None,
        hole_swatch=None,
        clear=False,
        **kwargs,
    ):
        """将任意 Manim 路径转换为具有"编织交错"效果的丝带图案

        孔洞的样式：

        - hole_opacity: 孔的透明度，默认 1
        - hole_color: 孔的颜色，默认分组着色
        - hole_swatch: 孔的颜色列表，默认随机

        丝带的样式采用 VMobject 的关键字参数

        - clear: 是否移除输入对象的重复路径，包括反向路径，默认 False
        """
        if DEV_MODE:
            print("=========== 开始 PROFILING ===========")
        super().__init__(**kwargs)
        self.offset = offset
        self.atol = atol
        self.clear_input_path = clear
        self.kwargs = kwargs

        # 输出容器
        self.ribbons = VGroup(**kwargs)
        self.holes = VGroup()

        # 内部数据
        self.polylines: list[Polyline] = []
        self.parallel_polylines: list[ParallelPolyline] = []
        self.d_anchors: dict[int, Anchor] = {}
        self.d_connections: dict[frozenset, Anchor] = {}
        self.overlaps: list[Overlap] = []
        self.hole_fragments: list[HoleFragment] = []
        self.ribbon_fragments: list[RibbonFragment] = []

        # 初始化
        self._init_polylines(target_mobject)
        if not self.polylines:
            return

        self._set_anchors_and_links()
        self._set_overlaps()
        self._set_hole_fragments()
        self._set_z_index()  # 一定要在 _set_ribbon_fragments 之前调用
        self._set_ribbon_fragments()
        self._generate_visuals(hole_opacity, hole_color, hole_swatch)

        self.add(
            self.holes,
            self.ribbons,
        )
        if DEV_MODE:
            self._generate_debug_visuals()

    def _init_polylines(self, mobject: VMobject):
        """从 mobject 提取多边形/折线"""
        paths = extract_paths(mobject, self.atol, self.clear_input_path)
        if DEV_MODE:
            min_dist = compute_global_min_distance(paths)
            if min_dist < self.offset * 2:
                print(f"[INFO] 全局最小距离: {min_dist:.4f}, 建议缩小 offset 至 {min_dist / 2:.4f}")

        for path in paths:
            # 判断是否闭合
            if np.allclose(path[0], path[-1]):
                poly = Polyline(path[:-1], True)
            else:
                poly = Polyline(path, False)
            self.polylines.append(poly)
            ppoly = ParallelPolyline(poly, self.offset)
            self.parallel_polylines.append(ppoly)

    @timer
    def _set_anchors_and_links(self):
        """计算所有交点"""
        offset_segs = self._get_offset_segs()
        # 计算 offset segments 的交点
        calc_all_intersections(offset_segs, self.d_anchors, self.d_connections)

        # 为 offset segments 创建 links
        for seg in offset_segs:
            anchor_pairs = list(zip(seg.anchors, seg.anchors[1:]))
            for i, (start, end) in enumerate(anchor_pairs):
                link = Link(start, end)
                # 关键：按索引奇偶性标记 is_overlap
                if i % 2 == 1:
                    link.is_overlap = True
                seg.links.append(link)

        """为两条偏移线的对应 links 设置 twin"""
        for ppoly in self.parallel_polylines:
            poly1, poly2 = ppoly.offset_polylines
            links1 = list(poly1.iter_links())
            links2 = list(poly2.iter_links())
            for i, link1 in enumerate(links1):
                if i < len(links2):
                    link2 = links2[i]
                    link1.twin = link2
                    link2.twin = link1

    @timer
    def _set_overlaps(self):
        nx_graph = networkx.Graph()
        for link in self._iter_offset_links():
            if link.is_overlap:
                nx_graph.add_edge(link.start.id, link.end.id, link=link)

        cycles = networkx.cycle_basis(nx_graph)
        for cycle in cycles:
            cycle.append(cycle[0])
            edges = list(zip(cycle, cycle[1:]))
            links: list[Link] = [nx_graph.edges[edge]["link"] for edge in edges]
            overlap = Overlap()
            overlap.links = links

            for link in links:
                link.overlap = overlap
            self.overlaps.append(overlap)

    @timer
    def _set_hole_fragments(self):
        """
        利用 twin 属性剔除包裹编织带的本体区域，并智能缝合边缘开口
        """
        nx_graph = networkx.Graph()
        # 构建图，仅使用非 overlap 的线段
        for link in self._iter_offset_links():
            if not link.is_overlap:
                nx_graph.add_edge(link.start.id, link.end.id, link=link)

        self._close_open_holes(nx_graph)

        cycles = networkx.cycle_basis(nx_graph)

        for cycle in cycles:
            cycle_coords = np.array([self.d_anchors[x_id].coord for x_id in cycle])
            path_poly = MatplotlibPath(cycle_coords)

            edge_key = (cycle[0], cycle[1])
            if edge_key not in nx_graph.edges:
                edge_key = (cycle[1], cycle[0])
            link: Link = nx_graph.edges[edge_key]["link"]

            if link.twin:
                t_start = link.twin.start.coord
                t_end = link.twin.end.coord
                twin_midpoint = (t_start + t_end) / 2
                if path_poly.contains_point(twin_midpoint, radius=1e-6):
                    continue

            if not right_handed(cycle_coords):
                cycle_coords = cycle_coords[::-1]

            fragment = HoleFragment(cycle_coords)
            self.hole_fragments.append(fragment)

    @timer
    def _set_z_index(self):
        """
        计算 over/under，遍历 offset polyline 的 links，按索引奇偶性设置 z_index
        """

        def next_poly(exclude):
            for ppoly in self.parallel_polylines:
                poly1, poly2 = ppoly.offset_polylines
                if poly1 in exclude:
                    continue
                ind = 0
                for link in poly1.iter_links():
                    if link.is_overlap:
                        if link.overlap and link.overlap.drawable and link.overlap.visited:
                            even_odd = ind % 2 == 1
                            return (poly1, poly2, even_odd)
                        ind += 1
            return (None, None, None)

        for ppoly in self.parallel_polylines:
            poly1, poly2 = ppoly.offset_polylines
            exclude = []
            even_odd = 0

            while poly1:
                ind = 0
                for i, segments in enumerate(poly1.segments):
                    for j, link in enumerate(segments.links):
                        if link.overlap is None:
                            continue
                        link.overlap.visited = True
                        if ind % 2 == even_odd:
                            if link.overlap.drawable:
                                link.overlap.drawable = False
                                link.z_index = 1
                                # 同时标记 twin
                                if i < len(poly2.segments) and j < len(poly2.segments[i].links):
                                    link2 = poly2.segments[i].links[j]
                                    link2.z_index = 1
                        ind += 1
                exclude.extend([poly1, poly2])
                poly1, poly2, even_odd = next_poly(exclude)

    @timer
    def _set_ribbon_fragments(self):
        """
        收集所有 z_index == 0 的 links, 找闭合环
        """
        anchor_pairs: list[tuple[Anchor, Anchor]] = []

        # 收集 merged segments（非 over 的连续 segments）
        for segs in self._iter_offset_segs():
            chains = self._merged_segments(segs)
            for chain in chains:
                if len(chain) >= 2:
                    anchor_pairs.append((chain[0], chain[-1]))

        # 连接开放线的端点
        for ppoly in self.parallel_polylines:
            if not ppoly.polyline.closed:
                poly1, poly2 = ppoly.offset_polylines
                if poly1.segments and poly2.segments:
                    p1_start = poly1.segments[0].anchors[0] if poly1.segments[0].anchors else None
                    p1_end = poly1.segments[-1].anchors[-1] if poly1.segments[-1].anchors else None
                    p2_start = poly2.segments[0].anchors[0] if poly2.segments[0].anchors else None
                    p2_end = poly2.segments[-1].anchors[-1] if poly2.segments[-1].anchors else None

                    if p1_start and p2_start:
                        anchor_pairs.append((p1_start, p2_start))
                    if p1_end and p2_end:
                        anchor_pairs.append((p1_end, p2_end))

        # 收集 z_index == 0 的 overlap segments
        for link in self._iter_offset_links():
            if link.z_index == 0 and link.is_overlap:
                anchor_pairs.append((link.start, link.end))

        # 构建图并找环
        graph_edges = [(r[0].id, r[1].id) for r in anchor_pairs]

        nx_graph = networkx.Graph()
        nx_graph.add_edges_from(graph_edges)
        cycles = networkx.cycle_basis(nx_graph)
        if cycles:
            for cycle in cycles:
                cycle.append(cycle[0])

        if not cycles:
            return

        for cycle in cycles:
            cycle_edges = list(zip(cycle, cycle[1:]))
            # 重新排列成连续路径
            plait = [cycle_edges[0][0], cycle_edges[0][1]]
            dup = cycle_edges[1:]
            for _ in range(len(cycle_edges) - 1):
                last = plait[-1]
                found = False
                for edge in dup:
                    if edge[0] == last:
                        plait.append(edge[1])
                        dup.remove(edge)
                        found = True
                        break
                    if edge[1] == last:
                        plait.append(edge[0])
                        dup.remove(edge)
                        found = True
                        break
                if not found:
                    break

            anchors = [self.d_anchors[x] for x in plait if x in self.d_anchors]
            if len(anchors) < 3:
                continue

            coords = np.array([x.coord for x in anchors])
            if not right_handed(coords):
                coords = coords[::-1]

            fragment = HoleFragment(coords)
            self.ribbon_fragments.append(fragment)

    @functools.cached_property
    def fragments_group(self) -> dict[float, list[HoleFragment]]:
        if not self.hole_fragments:
            return {}

        # 1. 向量化计算所有质心
        coords = np.array([np.mean(f.coords, axis=0) for f in self.hole_fragments])
        global_center = np.mean(coords, axis=0)

        # 2. 使用欧几里得距离 (L2 Norm)
        # 这保证了方形晶格中：中心块(0) < 边块(1) < 拐角块(1.414)
        dists = np.linalg.norm(coords - global_center, axis=1)

        # 3. 排序
        sort_indices = np.argsort(dists)
        sorted_dists = dists[sort_indices]

        # 4. 智能阈值计算
        diffs = np.diff(sorted_dists)

        # 找到所有真实的距离跳跃（过滤掉 1e-6 以下的数值噪声）
        significant_gaps = diffs[diffs > 1e-5]

        if len(significant_gaps) > 0:
            # 关键：对于晶格，我们取最小跳跃幅度的 0.5 倍作为阈值
            auto_tolerance = np.min(significant_gaps) * 0.5
        else:
            auto_tolerance = 1e-3

        # 5. 分组逻辑
        breaks = np.where(diffs > auto_tolerance)[0] + 1
        frag_array = np.array(self.hole_fragments)[sort_indices]

        grouped_frags = np.split(frag_array, breaks)
        grouped_dists = np.split(sorted_dists, breaks)

        # 6. 返回结果：Key 为该组的代表性距离
        return {float(np.mean(d)): f.tolist() for d, f in zip(grouped_dists, grouped_frags)}

    @property
    def main_path_info(self):
        """
        获取主路径信息，包含原始顶点和带有 Z-order 的交叉点。
        返回结构: List[List[dict]]
        每个 dict 包含:
        - type: 'point' | 'crossing'
        - coord: np.array (坐标)
        - z_index: int (仅 crossing 有效, 1=Over, 0=Under)
        - overlap: Overlap (仅 crossing 有效)
        """
        result = []
        for ppoly in self.parallel_polylines:
            path_nodes = []
            # 获取主路径对应的偏移路径之一（用于提取 Link/Overlap 信息）
            # 因为 offset_polylines[0] 和 offset_polylines[1] 拓扑结构相同，取第一个即可
            ref_offset_poly = ppoly.offset_polylines[0]

            # 遍历每一段 segment
            for i, seg in enumerate(ppoly.polyline.segments):
                # 添加 segment 起点
                # 注意：如果是闭合路径或第一段，需要添加起点。
                # 为简化逻辑，我们在每一段添加起点。最后一段结束后如果需要闭合，通常由调用者处理或添加终点。
                # 这里我们添加 segment 的起点 (对应原始锚点)
                path_nodes.append({"type": "point", "coord": seg.coord1, "z_index": None})

                if i >= len(ref_offset_poly.segments):
                    break

                # 获取对应的偏移路径 segment
                offset_seg = ref_offset_poly.segments[i]
                # 遍历该 segment 上的 links，寻找 overlaps
                for link in offset_seg.links:
                    if link.is_overlap and link.overlap:
                        # 这是一个交叉点
                        path_nodes.append(
                            {
                                "type": "crossing",
                                "coord": link.overlap.center,
                                "z_index": 1 - link.z_index,  # 直接复用 link 的 z_index
                                "overlap": link.overlap,
                            }
                        )

            # 对于非闭合路径，添加最后一个点
            if not ppoly.polyline.closed:
                path_nodes.append({"type": "point", "coord": ppoly.polyline.coords[-1], "z_index": None})

            result.append(path_nodes)
        return result

    def _generate_debug_visuals(self):
        """生成调试可视化：主路径、原始点、交叉点"""
        from manim import Dot, Line, Text, PURE_RED, PURE_BLUE

        debug_group = VGroup()

        for path_idx, path_nodes in enumerate(self.main_path_info):
            # 1. 绘制连线
            points = [node["coord"] for node in path_nodes]
            if len(points) > 1:
                points_3d = [np.append(p, 0) for p in points]
                # 绘制线段
                for i in range(len(points_3d) - 1):
                    line = Line(points_3d[i], points_3d[i + 1], color=RED, stroke_opacity=0.5, stroke_width=1)
                    debug_group.add(line)
                # 如果是闭合路径，连接首尾
                # 这里简单判断一下首尾距离
                if np.linalg.norm(points[0] - points[-1]) < 1e-6:
                    line = Line(points_3d[-1], points_3d[0], color=RED, stroke_opacity=0.5, stroke_width=1)
                    debug_group.add(line)

            # 2. 绘制节点
            for i, node in enumerate(path_nodes):
                coord_3d = np.append(node["coord"], 0)
                if node["type"] == "point":
                    # 原始点：半透明黑色小点
                    dot = Dot(coord_3d, color=BLACK, fill_opacity=1, radius=0.05)
                    if path_idx == 0:
                        text = Text(str(i), color=BLACK, font_size=17, font="Consolas")
                        text.move_to(dot.get_center() + np.array([0, 0.15, 0]))
                        debug_group.add(text)
                    debug_group.add(dot)
                elif node["type"] == "crossing":
                    # 交叉点：根据 z_index 着色
                    if node["z_index"] == 1:  # Over
                        color = RED
                        radius = 0.05
                        opacity = 1
                        z_index = 1

                    else:  # Under
                        color = BLUE
                        radius = 0.08
                        opacity = 0.8
                        z_index = 0
                    dot = Dot(coord_3d, color=color, fill_opacity=opacity, radius=radius, z_index=z_index)
                    if path_idx == 0:
                        text = Text(
                            str(i), color=PURE_RED if z_index == 1 else PURE_BLUE, font_size=17, font="Consolas"
                        )
                        text.move_to(dot.get_center() + np.array([0, 0.15 * (1 if z_index == 1 else -1), 0]))
                        debug_group.add(text)
                    debug_group.add(dot)

        self.add(debug_group)

    @timer
    def _generate_visuals(self, hole_opacity, hole_color, hole_swatch):
        """生成可视化对象"""
        # 创建 hole_fragments 的可视化
        swatch = hole_swatch or random_swatch()
        num_groups = len(self.fragments_group.keys())
        swatch_len = len(swatch)
        for i, fragments in enumerate(self.fragments_group.values()):
            # 核心逻辑：计算映射索引
            if num_groups > 1:
                # 将 i 从 [0, num_groups-1] 线性映射到 [0, swatch_len-1]
                color_index = int(round(i * (swatch_len - 1) / (num_groups - 1)))
            else:
                # 只有一组时，默认取第0个，或者你可以改为取中间 len//2
                color_index = 0

            # 确保不越界（双重保险）
            color = swatch[color_index % swatch_len]

            for fragment in fragments:
                pts_3d = np.hstack([fragment.coords, np.zeros((len(fragment.coords), 1))])
                poly = Polygon(
                    *pts_3d,
                    stroke_width=0,
                    fill_opacity=hole_opacity,
                    fill_color=hole_color or color,
                )
                self.holes.add(poly)

        # 创建 ribbon_fragments 的可视化
        for fragment in self.ribbon_fragments:
            pts_3d = np.hstack([fragment.coords, np.zeros((len(fragment.coords), 1))])
            poly = Polygon(
                *pts_3d,
                color=self.kwargs.get("stroke_color", WHITE),
                **self.kwargs,
            )
            self.ribbons.add(poly)

    # ============================================================
    # 辅助方法
    # ============================================================

    def _get_main_segs(self) -> list[PolyLineSegment]:
        segs = []
        for ppoly in self.parallel_polylines:
            segs.extend(ppoly.polyline.segments)
        return segs

    def _get_offset_segs(self) -> list[PolyLineSegment]:
        segs = []
        for ppoly in self.parallel_polylines:
            for poly in ppoly.offset_polylines:
                segs.extend(poly.segments)
        return segs

    def _iter_offset_segs(self):
        for ppoly in self.parallel_polylines:
            for poly in ppoly.offset_polylines:
                yield from poly.segments

    def _iter_offset_links(self):
        for ppoly in self.parallel_polylines:
            for poly in ppoly.offset_polylines:
                for seg in poly.segments:
                    yield from seg.links

    def _close_open_holes(self, nx_graph: networkx.Graph):
        """
        缝合所有开放的孔（即没有形成完整循环的孔）
        """
        # 1. 找出图中所有“断头”节点 (度数为1)
        cap_ids = [n for n, d in nx_graph.degree() if d == 1]
        valid_caps: list[Anchor] = []

        # 建立 ID 到 Anchor 的反查，确认为 cap
        for nid in cap_ids:
            anchor = self.d_anchors.get(nid)
            if anchor is not None and anchor.is_cap:
                valid_caps.append(anchor)

        if len(valid_caps) > 1:
            coords = np.array([cap.coord for cap in valid_caps])
            # 使用 KDTree，查找最近的 k 个邻居
            tree = cKDTree(coords)
            # k >= 3 的原因：第1个是自己，第2个可能是"自己丝带的另一端"(要跳过)，第3个才是"隔壁丝带"(要连接)
            dists, indices = tree.query(coords, k=min(len(valid_caps), 5))

            processed_pairs = set()

            idx_list: list[int] = []
            for i, (_, idx_list) in enumerate(zip(dists, indices)):
                u_cap = valid_caps[i]

                # 能够到达这里的 u_cap 度数必定为 1，所以 edge 只有一个
                u_edges = list(nx_graph.edges(u_cap.id))
                if not u_edges:
                    continue

                # 获取该 link 对象
                u_edge_key = u_edges[0]  # (u_id, v_id)
                u_link: Link = nx_graph.edges[u_edge_key]["link"]

                # 找到该 link 的 twin link 的端点
                # twin 的端点就是“同一条丝带的另一条边”的端点 -> 这是我们要避免连接的！
                avoid_coords = []
                if u_link.twin:
                    avoid_coords.append(u_link.twin.start.coord)
                    avoid_coords.append(u_link.twin.end.coord)

                # --- 遍历邻居，寻找合适的配对 ---
                found_match = False
                for neighbor_idx in idx_list:
                    # 跳过无效索引（当 k > 点数量时会出现）
                    if neighbor_idx == len(valid_caps):
                        continue

                    # 1. 跳过自己
                    if i == neighbor_idx:
                        continue

                    target_cap = valid_caps[neighbor_idx]

                    # 2. 跳过孪生 link 的端点
                    # 使用坐标距离判断是否是需要避开的点 (容差 1e-4)
                    is_twin_cap = False
                    for ac in avoid_coords:
                        if np.linalg.norm(target_cap.coord - ac) < 1e-4:
                            is_twin_cap = True
                            break
                    if is_twin_cap:
                        continue

                    # 3. 找到合法的“隔壁邻居”，进行缝合
                    pair_key = tuple(sorted((u_cap.id, target_cap.id)))
                    if pair_key in processed_pairs:
                        found_match = True  # 对方已经连过我了，算匹配成功
                        break

                    # 创建虚拟连接
                    virtual_link = Link(u_cap, target_cap)
                    nx_graph.add_edge(u_cap.id, target_cap.id, link=virtual_link)
                    processed_pairs.add(pair_key)
                    found_match = True
                    break  # 一个端点只连一条线

                # 4. 如果没有找到匹配
                if not found_match and DEV_MODE:
                    print(f"存在开放端点 {u_cap.coord} 没有形成孔洞")

    def _merged_segments(self, segment: PolyLineSegment):
        """合并 segment 中 z_index == 0 的连续 links"""
        chains: list[list[Anchor]] = []
        if not segment.anchors:
            return chains

        chain: list[Anchor] = [segment.anchors[0]]
        for link in segment.links:
            if link.z_index == 0:
                if link.start.id == chain[-1].id:
                    chain.append(link.end)
                else:
                    chains.append(chain)
                    chain = [link.start, link.end]
        if chain not in chains:
            chains.append(chain)
        return chains


# ============================================================
# 几何工具函数
# 命名约定:
#   - coords 表示坐标数组，(n, 2) 的数组，n 是点的数量
#   - coord 表示单个坐标点，(2,)
# ============================================================


@timer
def extract_paths(mobject: VMobject, atol=1e-6, clear=True):
    paths: list[NDArray[np.float64]] = []
    mobjects = mobject.family_members_with_points()
    for mob in mobjects:
        subpaths = mob.get_subpaths()
        for subpath in subpaths:
            coords = subpath[:, :2]
            if len(coords) < 2:
                continue
            # 移除连续重复点
            mask = np.ones(len(coords), dtype=bool)
            mask[1:] = np.linalg.norm(coords[1:] - coords[:-1], axis=1) > atol
            coords = coords[mask]
            if len(coords) > 2:
                # 移除共线点
                collinear_mask = np.ones(len(coords), dtype=bool)
                v1 = coords[1:-1] - coords[:-2]
                v2 = coords[2:] - coords[1:-1]
                cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                collinear_mask[1:-1] = np.abs(cross) >= atol
                coords = coords[collinear_mask]
            if len(coords) > 1:
                paths.append(coords)
    if clear:
        paths = remove_duplicate_paths(paths, atol=atol)
    return paths


@timer
def remove_duplicate_paths(paths: list[NDArray[np.float64]], atol=1e-4):
    """移除重复路径,包括首尾颠倒的"""
    if len(paths) <= 1:
        return paths

    # 按路径点数分组
    length_groups: dict[int, list[int]] = {}
    for i, path in enumerate(paths):
        n = len(path)
        if n not in length_groups:
            length_groups[n] = []
        length_groups[n].append(i)

    # 为每个长度组单独处理
    for n, indices in length_groups.items():
        if n < 2:
            continue
        used = [False] * len(indices)
        for i in range(len(indices)):
            if used[i]:
                continue
            for j in range(i + 1, len(indices)):
                if used[j]:
                    continue
                path_i = paths[indices[i]]
                path_j = paths[indices[j]]

                # 检查 path_j 是否与 path_i 相同（正序或倒序）
                if np.allclose(path_i, path_j, atol=atol) or np.allclose(path_i, path_j[::-1], atol=atol):
                    used[j] = True
                    break

        # 标记需要移除的路径
        for idx, is_used in zip(indices, used):
            if is_used:
                paths[idx] = None

    return [p for p in paths if p is not None]


def create_manim_graph(coords: NDArray[np.float64], labels=False):
    """从坐标数组创建 Manim 图形,一条带节点小圆的连线"""

    closed = np.allclose(coords[0], coords[-1])
    if closed:
        coords = coords[:-1]
        vertices_index = range(len(coords))
        edges = list(zip(vertices_index, np.roll(vertices_index, -1)))
    else:
        vertices_index = range(len(coords))
        edges = list(zip(vertices_index, vertices_index[1:]))

    coords_3d = np.hstack([coords, np.zeros((len(coords), 1))])
    return Graph(
        vertices_index,
        edges,
        layout=dict(zip(vertices_index, coords_3d)),
        vertex_config={"radius": 0.05, "color": RED},
        edge_config={"stroke_width": 4, "stroke_opacity": 0.5, "color": BLUE},
        labels=labels,
    )


def cal_segments_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """检查两条线段是否相交"""
    x2_x1 = x2 - x1
    y2_y1 = y2 - y1
    x4_x3 = x4 - x3
    y4_y3 = y4 - y3
    denom = (y4_y3) * (x2_x1) - (x4_x3) * (y2_y1)

    if np.isclose(denom, 0):
        return None

    x1_x3 = x1 - x3
    y1_y3 = y1 - y3
    ua = ((x4_x3) * (y1_y3) - (y4_y3) * (x1_x3)) / denom

    if ua < 0 or ua > 1:
        return None

    ub = ((x2_x1) * (y1_y3) - (y2_y1) * (x1_x3)) / denom

    if ub < 0 or ub > 1:
        return None

    x = x1 + ua * (x2_x1)
    y = y1 + ua * (y2_y1)
    return np.array([x, y])


@timer
def calc_all_intersections(
    segments: list[PolyLineSegment],
    d_anchors: dict[int, Anchor],
    d_connections: dict[frozenset, Anchor],
):
    """
    计算所有 segments 之间的交点，扫描线算法
    """
    # 注册已有的端点交点
    for seg in segments:
        for x in seg.anchors:
            d_anchors[x.id] = x

    if not segments:
        return

    # 构建 segment 数组
    seg_arr = np.array([[seg.coord1[0], seg.coord1[1], seg.coord2[0], seg.coord2[1], seg.id] for seg in segments])
    seg_arr_num = len(seg_arr)

    # 预计算边界框
    xmin = np.minimum(seg_arr[:, 0], seg_arr[:, 2]).reshape(seg_arr_num, 1)
    xmax = np.maximum(seg_arr[:, 0], seg_arr[:, 2]).reshape(seg_arr_num, 1)
    ymin = np.minimum(seg_arr[:, 1], seg_arr[:, 3]).reshape(seg_arr_num, 1)
    ymax = np.maximum(seg_arr[:, 1], seg_arr[:, 3]).reshape(seg_arr_num, 1)
    seg_arr = np.concatenate((seg_arr, xmin, ymin, xmax, ymax), axis=1)

    d_segments = {seg.id: seg for seg in segments}
    i_id, i_xmin, i_ymin, i_xmax, i_ymax = 4, 5, 6, 7, 8

    # 按 xmin 排序
    seg_arr = seg_arr[seg_arr[:, i_xmin].argsort()]

    for i in range(seg_arr_num):
        x1, y1, x2, y2, id1, sl_xmin, sl_ymin, sl_xmax, sl_ymax = seg_arr[i, :]
        seg1_vertices = [x1, y1, x2, y2]
        start = i + 1

        # 边界框过滤
        candidates = seg_arr[start:, :][
            ((seg_arr[start:, i_xmax] >= sl_xmin) & (seg_arr[start:, i_xmin] <= sl_xmax))
            & ((seg_arr[start:, i_ymax] >= sl_ymin) & (seg_arr[start:, i_ymin] <= sl_ymax))
        ]

        for candid in candidates:
            id2 = candid[i_id]
            seg2_vertices = candid[:4]
            x_point = cal_segments_intersection(*seg1_vertices, *seg2_vertices)

            if x_point is not None:
                seg1, seg2 = d_segments[int(id1)], d_segments[int(id2)]
                key = frozenset((seg1.id, seg2.id))

                if key not in d_connections:
                    anchor = Anchor(x_point, seg1, seg2)
                    d_anchors[anchor.id] = anchor
                    d_connections[key] = anchor
                    seg1.anchors.append(anchor)
                    seg2.anchors.append(anchor)

    # 排序每个 path 的交点
    for seg in segments:
        seg._sort_anchors()


def right_handed(polygon_coords: NDArray[np.float64]) -> bool:
    """判断多边形是否为逆时针方向"""
    coords = np.asarray(polygon_coords)
    if len(coords) < 3:
        return True

    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]

    x = coords[:, 0]
    y = coords[:, 1]
    area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return area > 0


def generate_offset_lines(coords: NDArray[np.float64], offset, closed):
    # 1. 预处理切线向量
    # 如果是闭合曲线，我们需要考虑从最后一个点回到第一个点的向量
    if closed:
        # 检查物理坐标是否已经闭合
        if np.allclose(coords[0], coords[-1]):
            # 如果已经闭合，np.diff 得到 N-1 个向量，对应 N-1 个顶点，
            # 但最后一个顶点和第一个重合，逻辑上还是 N-1 条边。
            # 为了通用性，我们去掉重复的末尾点进行计算，最后再补回去
            eff_coords = coords[:-1]
        else:
            eff_coords = coords

        # 使用 np.roll 计算相邻点差值，这样可以一次性得到所有边（包括闭合边）
        # axis=0 上的 roll(-1) 相当于把点序列向前移一位，相减即得到 p[i] -> p[i+1]
        tangents = np.roll(eff_coords, -1, axis=0) - eff_coords
    else:
        # 非闭合情况，直接计算
        tangents = np.diff(coords, axis=0)

    # 2. 计算法线 (Normals)
    norms = np.linalg.norm(tangents, axis=1)
    norms[norms < 1e-6] = 1.0
    tangents /= norms[:, None]

    # 旋转 90 度得到法线 (-y, x)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    # 3. 计算顶点偏移向量 (Miter Join)
    vn = np.zeros_like(coords)

    if closed:
        # 对于闭合图形，每个顶点的 n_in 是上一条边的法线，n_out 是当前边的法线
        # 使用 roll(1) 拿到“上一条边”的法线
        n_in = np.roll(normals, 1, axis=0)
        n_out = normals

        # 核心 Miter 公式
        denom = 1 + np.sum(n_in * n_out, axis=1)
        denom[denom < 1e-6] = 1e-6
        vn_body = (n_in + n_out) / denom[:, None]

        # 如果原始输入是物理闭合的（首尾点相同），我们需要把算好的第0个顶点的法线赋给最后一个点
        if np.allclose(coords[0], coords[-1]):
            vn[:-1] = vn_body
            vn[-1] = vn_body[0]  # 闭合点保持一致
        else:
            vn = vn_body

    else:
        # 非闭合情况：中间节点正常计算
        n_in = normals[:-1]
        n_out = normals[1:]

        denom = 1 + np.sum(n_in * n_out, axis=1)
        denom[denom < 1e-6] = 1e-6
        vn[1:-1] = (n_in + n_out) / denom[:, None]

        # 端点处理：直接使用端点边的法线（Butt cap 效果）
        vn[0] = normals[0]
        vn[-1] = normals[-1]

    # 4. Miter Limit 限制 (防止尖角无限长)
    limit = 5.0
    lens = np.linalg.norm(vn, axis=1)
    # 只有当长度真的超过限制才截断
    mask = lens > limit
    if np.any(mask):
        vn[mask] *= limit / lens[mask][:, None]

    return coords + vn * offset, coords - vn * offset


@timer
def compute_global_min_distance(paths: list[NDArray[np.float64]]):
    """计算所有路径中与其他路径最小的距离"""
    # 1. 记录每个点属于哪条路径 (标签化)
    all_points = np.vstack(paths)
    labels = np.concatenate([np.full(len(p), i) for i, p in enumerate(paths)])

    # 2. 构建全局 KD-Tree
    tree = cKDTree(all_points)

    # 3. 初始搜索：查询每个点的最近 2 个邻居
    # dists[:, 0] 是点到自身的距离(0)，dists[:, 1] 是最近邻距离
    dists, indices = tree.query(all_points, k=2, workers=-1)

    min_dist = float("inf")

    # 4. 检查最近邻是否属于不同路径
    # labels[i] 是当前点路径 ID，labels[indices[i, 1]] 是最近邻路径 ID
    is_different_path = labels != labels[indices[:, 1]]

    if np.any(is_different_path):
        min_dist = np.min(dists[is_different_path, 1])

    # 5. 极端情况处理 (Slow Path):
    # 如果所有点的最近邻都在同一条路径内，我们需要增加 k 继续搜索
    # 但对于实际的路径数据，通常 k=2 或 k=5 就能找到不同路径的点
    k = 2
    while min_dist == float("inf") and k < len(all_points):
        k = min(k * 2, len(all_points))
        dists, indices = tree.query(all_points, k=k, workers=-1)
        for i in range(1, k):  # 检查第 i 个邻居
            mask = labels != labels[indices[:, i]]
            if np.any(mask):
                min_dist = min(min_dist, np.min(dists[mask, i]))
                break

    return float(min_dist)
