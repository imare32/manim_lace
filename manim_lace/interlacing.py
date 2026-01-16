import warnings
from collections import defaultdict

import networkx
import numpy as np
from scipy.spatial import cKDTree


def right_handed(coords):
    if len(coords) < 3:
        return True
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    x = coords[:, 0]
    y = coords[:, 1]
    area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return area > 0


def clear_paths(paths: list[np.ndarray], atol=1e-6):
    cleared = []
    for path in paths:
        coords = path[:, :2]
        if len(coords) < 2:
            continue
        dup_mask = np.ones(len(coords), dtype=bool)
        dup_mask[1:] = np.linalg.norm(coords[1:] - coords[:-1], axis=1) > atol
        coords = coords[dup_mask]
        if len(coords) > 2:
            colinr_mask = np.ones(len(coords), dtype=bool)
            v1 = coords[1:-1] - coords[:-2]
            v2 = coords[2:] - coords[1:-1]
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            colinr_mask[1:-1] = np.abs(cross) >= atol
            coords = coords[colinr_mask]
        if len(coords) > 1:
            cleared.append(coords)
    return cleared


def merge_paths(paths: list[np.ndarray], atol=1e-6):
    """
    Merges connected paths into longer paths or closed loops.
    """
    if not paths:
        return []
    decimals = int(np.log10(1 / atol))
    # 收集所有坐标
    coords_all = np.concatenate(paths, axis=0)
    coords_all = np.around(coords_all, decimals)

    # 预计算所有坐标的tuple键
    coord_tuples = [tuple(coord) for coord in coords_all]
    d_coord_node = {coord: i for i, coord in enumerate(coord_tuples)}

    # 构建边列表
    edges = []
    for path in paths:
        path_tuples = [tuple(np.around(coord, decimals)) for coord in path]
        path_indices = [d_coord_node[t] for t in path_tuples]
        for i in range(len(path_indices) - 1):
            edges.append((path_indices[i], path_indices[i + 1]))

    networkx_graph = networkx.Graph()
    networkx_graph.add_edges_from(edges)

    merged = []

    # 记录每个 path 的起始节点索引
    path_starts = []
    for path in paths:
        if len(path) > 0:
            start_tuple = tuple(np.around(path[0], decimals))
            if start_tuple in d_coord_node:
                path_starts.append(d_coord_node[start_tuple])

    # 处理环（cycle basis）
    cycles = networkx.cycle_basis(networkx_graph)
    for cycle in cycles:
        if len(cycle) < 3:
            continue

        # 尝试将 cycle 旋转到某个 path 的起始点
        target_start = next((ps for ps in path_starts if ps in cycle), None)

        if target_start is not None:
            start_idx = cycle.index(target_start)
            cycle = cycle[start_idx:] + cycle[:start_idx]

        cycle_closed = cycle + [cycle[0]]
        coord_list = [coords_all[idx] for idx in cycle_closed]
        coords = np.array(coord_list)
        if not right_handed(coords):
            coords = coords[::-1]
        merged.append(coords)

    # 处理开链
    for island in networkx.connected_components(networkx_graph):
        island_list = list(island)
        if is_cycle(networkx_graph, island_list):
            continue
        if is_open_walk(networkx_graph, island_list):
            edges_island = [(u, v) for u, v in networkx_graph.edges if u in island_list and v in island_list]
            nodes = longest_chain(edges_island)
            if nodes:
                coord_list = [coords_all[idx] for idx in nodes]
                coords = np.array(coord_list)
                if not right_handed(coords):
                    coord_list.reverse()
                merged.append(np.array(coord_list))

    return merged


def longest_chain(edges):
    if not edges:
        return []

    # 构建邻接表
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # 找到端点（度为1的节点）或起点
    start = None
    end = None
    for node, neighbors in adj.items():
        if len(neighbors) == 1:
            if start is None:
                start = node
            else:
                end = node
                break

    # 如果没有端点（是环），选任意节点
    if start is None:
        start = edges[0][0]

    # 双端遍历构建链
    chain = [start]
    visited_edges = set()

    # 向后遍历
    current = start
    while True:
        neighbors = adj.get(current, [])
        next_node = None
        for n in neighbors:
            edge = tuple(sorted((current, n)))
            if edge not in visited_edges:
                next_node = n
                visited_edges.add(edge)
                break
        if next_node is None:
            break
        chain.append(next_node)
        current = next_node
        if end and current == end:
            break

    # 检查是否闭合
    if len(chain) > 1 and chain[0] == chain[-1]:
        chain = chain[:-1]  # 移除重复的起点

    return chain


def is_cycle(graph: networkx.Graph, island):
    degrees = [graph.degree(node) for node in island]
    return set(degrees) == {2}


def is_open_walk(graph: networkx.Graph, island):
    if len(island) == 2:
        return True
    degrees = [graph.degree(node) for node in island]
    return set(degrees) == {1, 2} and degrees.count(1) == 2


def generate_offset_lines(coords: np.ndarray, offset=0.1):
    closed = np.allclose(coords[0], coords[-1])
    if closed:
        eff_coords = coords[:-1]
        tangents = np.roll(eff_coords, -1, axis=0) - eff_coords
    else:
        tangents = np.diff(coords, axis=0)

    norms = np.linalg.norm(tangents, axis=1)
    norms[norms < 1e-6] = 1.0
    tangents /= norms[:, None]

    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    vn = np.zeros_like(coords)

    if closed:
        n_in = np.roll(normals, 1, axis=0)
        n_out = normals
        denom = 1 + np.sum(n_in * n_out, axis=1)
        denom[denom < 1e-6] = 1e-6
        vn_body = (n_in + n_out) / denom[:, None]

        if np.allclose(coords[0], coords[-1]):
            vn[:-1] = vn_body
            vn[-1] = vn_body[0]
        else:
            vn = vn_body

    else:
        n_in = normals[:-1]
        n_out = normals[1:]

        denom = 1 + np.sum(n_in * n_out, axis=1)
        denom[denom < 1e-6] = 1e-6
        vn[1:-1] = (n_in + n_out) / denom[:, None]

        vn[0] = normals[0]
        vn[-1] = normals[-1]

    limit = 5.0
    lens = np.linalg.norm(vn, axis=1)
    mask = lens > limit
    if np.any(mask):
        vn[mask] *= limit / lens[mask][:, None]

    return coords + vn * offset, coords - vn * offset


def vectorized_intersection(P1, P2, P3, P4):
    d1, d2, d13 = P2 - P1, P4 - P3, P1 - P3
    denom = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]
    is_p = np.abs(denom) < 1e-9
    denom_s = np.where(is_p, 1.0, denom)
    ua = (d2[:, 0] * d13[:, 1] - d2[:, 1] * d13[:, 0]) / denom_s
    ub = (d1[0] * d13[:, 1] - d1[1] * d13[:, 0]) / denom_s
    mask = (~is_p) & (ua >= -1e-9) & (ua <= 1 + 1e-9) & (ub >= -1e-9) & (ub <= 1 + 1e-9)
    return P1 + ua[:, None] * d1, mask


class Interlacing:
    def __init__(self, paths: list[np.ndarray], offset=0.1):
        """
        初始化 Lace 类，计算路径的偏移线和交点。
        参数:
            paths (list[np.ndarray]): 输入的路径列表，每个路径是一个二维 numpy 数组。
        """
        self.paths = merge_paths(clear_paths(paths))
        if not self.paths:
            return
        self.paths_closed = []
        self.paths_flag = []
        # 排序的目的是为了保证开放路径的 flag 初始值是交替为 True 和 False
        self.paths.sort(key=lambda x: (x[0][0], x[0][1]))
        flag = True
        for path in self.paths:
            closed = np.allclose(path[0], path[-1])
            self.paths_closed.append(closed)
            if closed:
                # 确保闭合路径的首尾点相同, 消除误差
                path[-1] = path[0]
                self.paths_flag.append(True)
            else:
                self.paths_flag.append(flag)
                flag = not flag
        self.paths_l = []
        self.paths_r = []
        for path in self.paths:
            path_l, path_r = generate_offset_lines(path, offset)
            self.paths_l.append(path_l)
            self.paths_r.append(path_r)
        assert len(self.paths) == len(self.paths_l) == len(self.paths_r)
        self.d_segments = {}
        self.d_intersections = {}
        self.d_seg_intersects = defaultdict(list)
        for i, (path, path_l, path_r) in enumerate(zip(self.paths, self.paths_l, self.paths_r)):
            for j, (p, p_l, p_r) in enumerate(zip(path, path_l, path_r)):
                if j < len(path) - 1:
                    self.d_segments[(i, 0, j)] = (p, path[j + 1])
                    self.d_segments[(i, 1, j)] = (p_l, path_l[j + 1])
                    self.d_segments[(i, 2, j)] = (p_r, path_r[j + 1])

        self._cal_intersections()
        # self._set_overlaps()
        self._set_holes()
        self._set_ribbons()

    def _cal_intersections(self):
        seg_coords_list = []
        seg_ids_list = []

        # --- 1. 从 self.segments 提取数据 ---
        # 遍历字典，过滤掉 type_idx == 0 (中线)
        for (p_idx, t_idx, s_idx), (p_start, p_end) in self.d_segments.items():
            if t_idx == 0:
                continue

            # 构建坐标行: [x1, y1, x2, y2]
            coord_row = np.hstack((p_start, p_end))
            seg_coords_list.append(coord_row)
            seg_ids_list.append((p_idx, t_idx, s_idx))

        # 转换为 Numpy 数组进行向量化计算
        seg_arr = np.vstack(seg_coords_list)
        seg_ids_arr = np.array(seg_ids_list, dtype=object)
        num_segs = len(seg_arr)

        # --- 2. 构建 Augmented Segments 并排序 ---
        x1, y1, x2, y2 = seg_arr[:, 0], seg_arr[:, 1], seg_arr[:, 2], seg_arr[:, 3]
        xmin, xmax = np.minimum(x1, x2), np.maximum(x1, x2)
        ymin, ymax = np.minimum(y1, y2), np.maximum(y1, y2)

        # [x1, y1, x2, y2, xmin, ymin, xmax, ymax]
        aug_segs = np.column_stack((seg_arr, xmin, ymin, xmax, ymax))

        sort_idx = np.argsort(xmin)
        aug_segs = aug_segs[sort_idx]
        sorted_ids = seg_ids_arr[sort_idx]
        s_xmin = aug_segs[:, 4]

        intersection_clusters = defaultdict(set)

        # --- 3. 扫描线循环 ---
        for i in range(num_segs - 1):
            curr_seg = aug_segs[i]
            curr_id = tuple(sorted_ids[i])  # (p_idx, t_idx, s_idx)

            # 快速筛选 X 轴范围
            search_limit = np.searchsorted(s_xmin, curr_seg[6], side="right")
            if search_limit <= i + 1:
                continue

            candidates_idx = np.arange(i + 1, search_limit)

            # Y 轴范围过滤
            y_mask = (curr_seg[7] >= aug_segs[candidates_idx, 5]) & (curr_seg[5] <= aug_segs[candidates_idx, 7])

            valid_candidates_indices = candidates_idx[y_mask]
            if len(valid_candidates_indices) == 0:
                continue

            # 向量化求交
            P1, P2 = curr_seg[:2], curr_seg[2:4]
            cand_segs = aug_segs[valid_candidates_indices]
            P3, P4 = cand_segs[:, 0:2], cand_segs[:, 2:4]

            points, valid_mask = vectorized_intersection(P1, P2, P3, P4)

            if not np.any(valid_mask):
                continue

            # 处理有效交点
            for point, m_idx in zip(points[valid_mask], valid_candidates_indices[valid_mask]):
                target_id = tuple(sorted_ids[m_idx])

                # 记录
                self.d_intersections[frozenset((curr_id, target_id))] = point
                self.d_seg_intersects[curr_id].append(target_id)
                self.d_seg_intersects[target_id].append(curr_id)

                coord_key = (round(point[0], 6), round(point[1], 6))
                intersection_clusters[coord_key].add(curr_id)
                intersection_clusters[coord_key].add(target_id)

        # --- 4. 收尾 ---
        for coord, seg_set in intersection_clusters.items():
            if len(seg_set) >= 3:
                warnings.warn(f"Multi-segment intersection at {coord} involves {len(seg_set)} segments.", UserWarning)

        for seg_id in self.d_seg_intersects:
            seg_start = self.d_segments[seg_id][0]
            self.d_seg_intersects[seg_id].sort(
                key=lambda tgt_id: np.linalg.norm(self.d_intersections[frozenset((seg_id, tgt_id))] - seg_start),
            )

    def _set_overlaps(self):
        self.overlaps = []
        nx_graph = networkx.Graph()
        for seg_id, tgt_ids in self.d_seg_intersects.items():
            # 正常情况下, 交点数应该是偶数, 因为线段被双轨的偏移线切割
            # 0是线段的起点, 所以从1开始
            # 每两个线段之间只能有一个交点, 所以两个线段的编号决定一个交点或拓扑节点
            for k in range(1, len(tgt_ids) - 1, 2):
                nx_graph.add_edge(
                    frozenset((seg_id, tgt_ids[k])),
                    frozenset((seg_id, tgt_ids[k + 1])),
                )

        cycles = networkx.cycle_basis(nx_graph)
        for cycle in cycles:
            # 4个交叉点组成一个重叠区域
            if not len(cycle) == 4:
                continue
            self.overlaps.append(np.array([self.d_intersections[node] for node in cycle + [cycle[0]]]))

    def _set_holes(self):
        # 添加开放路径的端点, 这些端点也会被 _set_ribbons 用到
        for i, (paths_l, paths_r) in enumerate(zip(self.paths_l, self.paths_r)):
            if self.paths_closed[i]:
                continue
            num_seg = len(paths_l) - 1
            seg_id_l, tgt_id_l = (i, 1, 0), 0
            self.d_intersections[frozenset((seg_id_l, tgt_id_l))] = paths_l[0]
            self.d_seg_intersects[seg_id_l].insert(0, tgt_id_l)

            seg_id_l, tgt_id_l = (i, 1, num_seg - 1), -1
            self.d_intersections[frozenset((seg_id_l, tgt_id_l))] = paths_l[-1]
            self.d_seg_intersects[seg_id_l].append(tgt_id_l)

            seg_id_r, tgt_id_r = (i, 2, 0), 0
            self.d_intersections[frozenset((seg_id_r, tgt_id_r))] = paths_r[0]
            self.d_seg_intersects[seg_id_r].insert(0, tgt_id_r)

            seg_id_r, tgt_id_r = (i, 2, num_seg - 1), -1
            self.d_intersections[frozenset((seg_id_r, tgt_id_r))] = paths_r[-1]
            self.d_seg_intersects[seg_id_r].append(tgt_id_r)

        self.holes = []
        nx_graph = networkx.Graph()

        # 为开放路径的边界添加虚拟连接
        # === 1. 收集所有"断头"节点 (Valid Caps) ===
        cap_nodes = []  # 存储用于 networkx 的节点 ID (frozenset)
        cap_coords = []  # 存储用于 KDTree 的坐标 (x, y)
        cap_info = []  # 存储元数据用于判断是否为"孪生": (path_idx, side_idx)

        path_idxs = [i for i in range(len(self.paths)) if not self.paths_closed[i]]

        for i in path_idxs:
            num_seg = len(self.paths_l[i]) - 1

            # 定义该路径的4个端点配置
            # 格式: (seg_idx, end_point_flag, side_idx)
            # end_point_flag: 0 是起点, -1 是终点
            # side_idx: 1 是左, 2 是右
            endpoints = [
                (0, 0, 1),  # 左起点
                (0, 0, 2),  # 右起点
                (num_seg - 1, -1, 1),  # 左终点
                (num_seg - 1, -1, 2),  # 右终点
            ]

            for s_idx, e_flag, side in endpoints:
                # 构建完整的节点 ID，必须与 d_intersections 中的 key 一致
                node_id = frozenset(((i, side, s_idx), e_flag))

                if node_id in self.d_intersections:
                    coord = self.d_intersections[node_id]
                    cap_nodes.append(node_id)
                    cap_coords.append(coord)
                    cap_info.append((i, side, e_flag))  # 记录路径索引 i，用于判断 twin

        # === 2. 使用 KDTree 查找邻居并缝合 ===
        if len(cap_coords) > 0:
            tree = cKDTree(cap_coords)
            # k=3: 找3个最近的 (1个是自己，1个是孪生，1个是目标邻居)
            # 适当放大 k 以防万一，比如 k=4
            dists_all, idxs_all = tree.query(cap_coords, k=min(3, len(cap_coords)))

            added_edges = set()

            for i, (dists, idxs) in enumerate(zip(dists_all, idxs_all)):
                current_node = cap_nodes[i]
                current_path_idx = cap_info[i][0]
                current_eflag = cap_info[i][2]

                # 遍历找到的邻居
                valid_neighbor_found = False
                for d, neighbor_idx in zip(dists, idxs):
                    if i == neighbor_idx:
                        continue  # 跳过自己

                    neighbor_path_idx = cap_info[neighbor_idx][0]
                    neighbor_eflag = cap_info[neighbor_idx][2]

                    # === 关键逻辑: 排除孪生 ===
                    # 如果邻居属于同一条 Path (path_idx 相同)，则是"孪生轨道的端点"
                    # 我们不想连接同一条丝带的左右两边(那是丝带的封口)，我们想连隔壁的丝带
                    # 但如果是同一条路径的首尾相接 (Start -> End)，则是允许的 (形成闭环)
                    if current_path_idx == neighbor_path_idx and current_eflag == neighbor_eflag:
                        valid_neighbor_found = True
                        continue

                    # === 距离阈值保护 ===
                    # 防止连接到太远的地方 (比如画布另一头)
                    # 阈值可以设为 offset * 4 或者根据实际情况动态调整
                    # 这里假设如果超过 5.0 单位就不连了 (根据你的代码 generate_offset_lines 里的 limit 推测)
                    if d > 10.0:
                        continue

                    target_node = cap_nodes[neighbor_idx]

                    # 避免重复添加边 (A->B 和 B->A)
                    edge_sig = frozenset((current_node, target_node))
                    if edge_sig not in added_edges:
                        nx_graph.add_edge(current_node, target_node)
                        added_edges.add(edge_sig)
                        valid_neighbor_found = True

                    # 只要找到最近的一个合法邻居连接即可，通常不需要连多个
                    if valid_neighbor_found:
                        break
                if not valid_neighbor_found:
                    warnings.warn(f"Node {current_node} has no valid neighbors.", UserWarning)

        # === 3. 连接丝带内部的 edges ===
        for seg_id, tgt_ids in self.d_seg_intersects.items():
            for k in range(0, len(tgt_ids), 2):
                if k + 1 < len(tgt_ids):
                    nx_graph.add_edge(
                        frozenset((seg_id, tgt_ids[k])),
                        frozenset((seg_id, tgt_ids[k + 1])),
                    )

        # === 4. 计算 Loop 面积并提取 Holes ===
        cycles = networkx.cycle_basis(nx_graph)

        valid_cycle_data = []
        areas = []
        for cycle in cycles:
            if len(cycle) < 3:
                warnings.warn(f"Hole cycle has {len(cycle)} nodes.", UserWarning)
                continue

            coords = np.array([self.d_intersections[node] for node in cycle + [cycle[0]]])

            # Shoelace formula for area
            x = coords[:, 0]
            y = coords[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

            valid_cycle_data.append((coords, area))
            areas.append(area)

        areas.sort()
        total_area = sum(areas[:-1])
        print(len(areas))

        for coords, area in valid_cycle_data:
            if area > total_area:
                continue

            if not right_handed(coords):
                coords = coords[::-1]
            self.holes.append(coords)

    def _set_ribbons(self):
        self.ribbons = []
        for i, path in enumerate(self.paths):
            flag = self.paths_flag[i]
            nx_graph = networkx.Graph()
            num_seg = len(path) - 1

            # 开放路径的端点封口
            if not self.paths_closed[i]:
                seg_id_l = (i, 1, 0)
                seg_id_r = (i, 2, 0)
                nx_graph.add_edge(
                    frozenset((seg_id_l, 0)),
                    frozenset((seg_id_r, 0)),
                )
                seg_id_l = (i, 1, num_seg - 1)
                seg_id_r = (i, 2, num_seg - 1)
                nx_graph.add_edge(
                    frozenset((seg_id_l, -1)),
                    frozenset((seg_id_r, -1)),
                )

            for j in range(num_seg):
                seg_id_l = (i, 1, j)
                seg_id_r = (i, 2, j)
                tgt_ids_l = self.d_seg_intersects[seg_id_l]
                tgt_ids_r = self.d_seg_intersects[seg_id_r]

                if len(tgt_ids_l) < 2:
                    continue

                prev_node_l = frozenset((seg_id_l, tgt_ids_l[0]))
                prev_node_r = frozenset((seg_id_r, tgt_ids_r[0]))

                for k in range(2, len(tgt_ids_l) - 1, 2):
                    # flag 为 True 表示当前的 seg 会被 (k - 1, k) 这一组交点截断
                    # 需要在截断前添加三条 edges 组成一个u型, 在截断后添加一条 edge 封口
                    # flag 为 False 则延迟连接
                    if flag:
                        curr_start_l = frozenset((seg_id_l, tgt_ids_l[k - 2]))
                        curr_start_r = frozenset((seg_id_r, tgt_ids_r[k - 2]))

                        if prev_node_l != curr_start_l:
                            nx_graph.add_edge(prev_node_l, curr_start_l)
                        if prev_node_r != curr_start_r:
                            nx_graph.add_edge(prev_node_r, curr_start_r)

                        nx_graph.add_edge(
                            curr_start_l,
                            frozenset((seg_id_l, tgt_ids_l[k - 1])),
                        )
                        nx_graph.add_edge(
                            curr_start_r,
                            frozenset((seg_id_r, tgt_ids_r[k - 1])),
                        )
                        nx_graph.add_edge(
                            frozenset((seg_id_r, tgt_ids_r[k - 1])),
                            frozenset((seg_id_l, tgt_ids_l[k - 1])),
                        )

                        next_start_l = frozenset((seg_id_l, tgt_ids_l[k]))
                        next_start_r = frozenset((seg_id_r, tgt_ids_r[k]))
                        nx_graph.add_edge(next_start_l, next_start_r)

                        prev_node_l = next_start_l
                        prev_node_r = next_start_r

                    flag = not flag

                end_node_l = frozenset((seg_id_l, tgt_ids_l[-1]))
                end_node_r = frozenset((seg_id_r, tgt_ids_r[-1]))

                if prev_node_l != end_node_l:
                    nx_graph.add_edge(prev_node_l, end_node_l)
                if prev_node_r != end_node_r:
                    nx_graph.add_edge(prev_node_r, end_node_r)

            cycles = networkx.cycle_basis(nx_graph)
            for cycle in cycles:
                if len(cycle) < 3:
                    continue
                coords = np.array([self.d_intersections[node] for node in cycle + [cycle[0]]])
                if not right_handed(coords):
                    coords = coords[::-1]
                self.ribbons.append(coords)
