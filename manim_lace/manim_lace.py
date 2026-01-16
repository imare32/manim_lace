import functools
from collections import defaultdict

import manim
import numpy as np

from .interlacing import Interlacing
from .swatches import random_swatch


class Lace(manim.VGroup):
    def __init__(
        self,
        target_mobject: manim.VMobject,
        offset=0.1,
        hole_opacity=1.0,
        hole_color=None,
        hole_swatch=None,
        hole_group_method="by_area",
        **kwargs,
    ):
        """将任意 Manim 路径转换为具有"编织交错"效果的丝带图案

        孔洞的样式：

        - hole_opacity: 孔的透明度，默认 1
        - hole_color: 孔的颜色，默认分组着色
        - hole_swatch: 孔的颜色列表，默认随机
        - hole_group_method: 孔的分组方法，默认按面积分组
            - by_area: 按面积分组
            - by_distance: 按距离分组，根据孔的中心距离全局中心排序
            - by_hole_index: 按孔索引分组
            - by_vertex_count: 按顶点数量分组



        丝带的样式采用 VMobject 的关键字参数
        """
        super().__init__()
        self.offset = offset
        self.hole_group_method = hole_group_method
        self.kwargs = kwargs

        paths = []
        mobjects = target_mobject.family_members_with_points()
        for mob in mobjects:
            paths.extend(mob.get_subpaths())
        self.interlacing = Interlacing(paths, offset)

        self.ribbons = manim.VGroup()
        self.holes = manim.VGroup()
        self._generate_visuals(hole_opacity, hole_color, hole_swatch)
        self.add(
            self.holes,
            self.ribbons,
        )

    def _generate_visuals(self, hole_opacity, hole_color, hole_swatch):
        """生成可视化对象"""
        swatch = hole_swatch or random_swatch()
        num_groups = len(self.hole_groups.keys())
        swatch_len = len(swatch)
        for i, holes in enumerate(self.hole_groups.values()):
            # 核心逻辑：计算映射索引
            if num_groups > 1:
                # 将 i 从 [0, num_groups-1] 线性映射到 [0, swatch_len-1]
                color_index = int(round(i * (swatch_len - 1) / (num_groups - 1)))
            else:
                # 只有一组时，默认取第0个，或者你可以改为取中间 len//2
                color_index = 0

            # 确保不越界（双重保险）
            color = swatch[color_index % swatch_len]

            for hole in holes:
                pts_3d = np.hstack([hole, np.zeros((len(hole), 1))])
                poly = manim.Polygon(
                    *pts_3d,
                    stroke_width=0,
                    fill_opacity=hole_opacity,
                    fill_color=hole_color or color,
                )
                self.holes.add(poly)

        # 创建 ribbon_fragments 的可视化
        for ribbon in self.interlacing.ribbons:
            pts_3d = np.hstack([ribbon, np.zeros((len(ribbon), 1))])
            poly = manim.Polygon(
                *pts_3d,
                color=self.kwargs.get("stroke_color", manim.WHITE),
                **self.kwargs,
            )
            self.ribbons.add(poly)

        # self.add(self.draw_path(self.interlacing.paths_l[0]))
        # self.add(self.draw_path(self.interlacing.paths_r[0]))

    def draw_path(self, coords):
        pts_3d = np.hstack([coords, np.zeros((len(coords), 1))])
        return manim.VMobject(stroke_color=manim.BLACK).set_points_as_corners(pts_3d)

    @functools.cached_property
    def hole_groups(self):
        holes = self.interlacing.holes
        if not holes:
            return {}

        groups = defaultdict(list)

        if self.hole_group_method == "by_area":
            for hole in holes:
                # 1. 顶点数
                num_verts = len(hole)

                # 2. 面积 (Shoelace formula)
                x = hole[:-1, 0]
                y = hole[:-1, 1]
                area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

                # 3. 构建键 (顶点数, 面积保留4位小数以忽略浮点误差)
                key = (num_verts, np.round(area, 4))

                groups[key].append(hole)

            # 4. 按键排序以保持颜色分配的稳定性
            # 排序优先级: 顶点数 -> 面积
            sorted_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))

        elif self.hole_group_method == "by_vertex_count":
            for hole in holes:
                num_verts = len(hole) - 1
                groups[num_verts].append(hole)
            sorted_keys = sorted(groups.keys())

        elif self.hole_group_method == "by_hole_index":
            for i, hole in enumerate(holes):
                groups[i].append(hole)
            sorted_keys = sorted(groups.keys())

        elif self.hole_group_method == "by_distance":
            coords = np.array([np.mean(hole[:-1], axis=0) for hole in holes])
            global_center = np.mean(coords, axis=0)

            # 2. 使用欧几里得距离 (L2 Norm)
            dists = np.linalg.norm(coords - global_center, axis=1)

            # 3. 排序
            sort_indices = np.argsort(dists)
            sorted_dists = dists[sort_indices]
            sorted_holes = [holes[i] for i in sort_indices]

            # 4. 智能阈值计算
            diffs = np.diff(sorted_dists)

            # 找到所有真实的距离跳跃（过滤掉 1e-5 以下的数值噪声）
            significant_gaps = diffs[diffs > 1e-5]

            if len(significant_gaps) > 0:
                # 关键：对于晶格，我们取最小跳跃幅度的 0.5 倍作为阈值
                auto_tolerance = np.min(significant_gaps) * 0.5
            else:
                auto_tolerance = 1e-3

            # 5. 分组逻辑
            current_group_id = 0
            # 添加第一个孔到第一组
            groups[current_group_id].append(sorted_holes[0])

            for i in range(len(sorted_dists) - 1):
                if sorted_dists[i + 1] - sorted_dists[i] > auto_tolerance:
                    current_group_id += 1
                groups[current_group_id].append(sorted_holes[i + 1])
            
            sorted_keys = sorted(groups.keys())

        return {k: groups[k] for k in sorted_keys}
