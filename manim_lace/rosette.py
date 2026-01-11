import math
import typing

import numpy as np
from manim import Polygram

STAR_TYPE = typing.Literal["PA", "PO", "PR", "CA", "CR"]


def cal_kernels(n, level=2, star_type: STAR_TYPE = "PA"):
    """计算 Rosette 的基础几何骨架 (kernels)"""
    θ = np.pi / n

    # 1. 基础坐标与初始 kernel 计算
    x0, y0 = np.cos(θ), np.sin(θ)
    x1, y1 = np.cos(2 * θ), np.sin(2 * θ)
    x2, y2 = np.cos(3 * θ), np.sin(3 * θ)

    xk0 = x0 + y0 / np.tan(2 * θ)
    xk1 = x1 + y1 / np.tan(θ)
    kernel_0 = np.array([[x0, y0], [xk0, 0]])
    kernel_1 = np.array([[x1, y1], [xk1, 0]])

    if star_type[0] == "P":  # Parallel 类型
        xk21 = xk0 + y2 / np.tan(2 * θ)
        xk22 = xk21 + y2 / np.cos(θ)
        if star_type[1] == "A":
            xk23 = xk22 + y2 * np.tan(θ)
        elif star_type[1] == "O":
            xk23 = xk22 - y2 * np.tan(2 * θ)
        elif star_type[1] == "R":
            xk23 = xk22
        kernel_2 = np.array([[x2, y2], [xk22, y2], [xk23, 0]])
    elif star_type[0] == "C":  # Convergent 类型
        xk22 = x2 + (y2 - y0) / np.tan(θ)
        if star_type[1] == "R":
            xk23 = xk22
        elif star_type[1] == "A":
            xk23 = xk22 + y0 * np.tan(θ)
        kernel_2 = np.array([[x2, y2], [xk22, y0], [xk23, 0]])

    kernels = [kernel_0, kernel_1, kernel_2]

    # 2. 迭代计算更高阶的 kernel
    rot_matrix = np.array([[x0, -y0], [y0, x0]])

    for _ in range(3, level + 1):
        pre_kernel = kernels[-1]
        rotated_kernel = pre_kernel @ rot_matrix.T
        rot_p1, rot_p2, rot_p3 = rotated_kernel[:3]

        # 计算旋转后线段与 X 轴的交点
        rx2, ry2 = rot_p2
        rx3, ry3 = rot_p3
        px_x = rx2 - ry2 * (rx3 - rx2) / (ry3 - ry2)
        px = np.array([px_x, 0.0])

        kernels.append(np.array([rot_p1, rot_p2, px]))

    return kernels


def get_petals_by_kernel(kernel, n):
    """通过 kernel 镜像并旋转生成所有花瓣"""
    θ = np.pi / n
    petals = []
    for i in range(n):
        # 镜像合并生成单个花瓣
        flip_part = kernel[-2::-1] * np.array([1, -1])
        petal = np.vstack([kernel, flip_part])[::-1]

        # 旋转到对应位置
        angle = i * 2 * θ
        c, s = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[c, -s], [s, c]])
        petals.append(petal @ rot_matrix.T)
    return petals


def sort_petals(petals, level):
    """根据拓扑连接顺序重新排列花瓣"""
    n = len(petals)
    step = level + 1
    num_cycles = math.gcd(n, step)
    cycle_len = n // num_cycles

    idxs = []
    for i in range(num_cycles):
        current = i
        for _ in range(cycle_len):
            idxs.append(current)
            current = (current + step) % n

    return [petals[i] for i in idxs]


def merge_adjacent_petals(petals):
    """合并首尾相接的花瓣点集"""
    if not petals:
        return []

    merged_groups = []
    current_poly = petals[0]

    for i in range(1, len(petals)):
        next_petal = petals[i]
        if np.allclose(current_poly[-1], next_petal[0]):
            current_poly = np.vstack((current_poly, next_petal[1:]))
        else:
            merged_groups.append(current_poly)
            current_poly = next_petal

    merged_groups.append(current_poly)

    # Polygram 会自动闭合多边形，需要移除闭合的点
    final_result = []
    for poly in merged_groups:
        if np.allclose(poly[0], poly[-1]) and len(poly) > 2:
            final_result.append(poly[:-1])
        else:
            final_result.append(poly)

    return final_result


class StarRosette(Polygram):
    """基于星形图案的 Manim Rosette 对象"""

    def __init__(self, n, level=2, star_type: STAR_TYPE = "PA", **kwargs):
        self.n = n

        kernels = cal_kernels(n, level, star_type)
        kernel = kernels[level]

        raw_petals = get_petals_by_kernel(kernel, n)
        sorted_petals = sort_petals(raw_petals, level)
        merged_points_list_2d = merge_adjacent_petals(sorted_petals)

        # 转换为 Manim 所需的 3D 坐标并初始化
        vertex_groups = [np.pad(points, ((0, 0), (0, 1))) for points in merged_points_list_2d]
        super().__init__(*vertex_groups, **kwargs)
