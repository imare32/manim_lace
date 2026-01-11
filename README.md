# Lace 算法文档

## 概述

Lace (Interlacing) 算法用于将任意 Manim 路径转换为具有"编织交错"效果的丝带图案。当多条路径相交时，算法会自动计算交替的上下遮挡关系，模拟真实编织物的视觉效果。

## 算法流程详解

Lace 算法的核心思想是将几何问题转化为拓扑图论问题。通过构建路径的偏移线（Offset Lines），识别交叉区域（Overlaps）和孔洞区域（Holes），并在图上进行状态传播来确定遮挡关系。

### 1. 预处理 (Preprocessing)

输入为任意的 Manim `VMobject`。

1.  **路径提取 (`extract_paths`)**:
    *   遍历输入对象的所有子路径。
    *   **去重**: 移除连续的重复坐标点。
    *   **去共线**: 移除三点共线中间的多余点，简化路径。
    *   **路径去重**: 移除几何上完全相同的路径（包括反向路径）。
    *   **KD-Tree 优化**: 计算所有路径之间的全局最小距离，用于建议合适的丝带宽度（offset）。

### 2. 偏移线生成 (Offset Generation)

为了模拟有宽度的丝带，算法需要为主路径生成两条平行的“偏移线”。

*   **算法**: 使用 **Miter Join** (斜接) 算法。
*   **法线计算**: 计算路径上每条线段的法向量。
*   **顶点偏移**: 对于每个顶点，计算相邻两条边法向量的角平分线方向，作为偏移方向。
*   **闭合处理**: 自动检测路径闭合性，确保闭合路径的偏移线首尾平滑连接。
*   **Miter Limit**: 防止在极锐角处产生过长的尖角，进行截断处理。

生成结果存储在 `ParallelPolyline` 对象中，包含一条主线 (`polyline`) 和两条偏移线 (`offset_polylines`)。

### 3. 构建拓扑图 (Topology Construction)

算法构建了一个基于 **Anchor (锚点)** 和 **Link (连线)** 的网络。

1.  **计算交点 (`_set_anchors_and_links`)**:
    *   使用 **扫描线算法 (Sweep-line Algorithm)** 高效计算所有偏移线段之间的交点。
    *   **Anchor**: 既包括原始路径的端点，也包括新计算出的交点。
    *   **Link**: 两个相邻 Anchor 之间的线段。
    *   **Twin Link**: 记录每条偏移线上的 Link 对应的另一侧偏移线上的 Link。

2.  **识别交叉区域 (`_set_overlaps`)**:
    *   在偏移线的交织网络中，交叉区域表现为一个由 4 条 Link 组成的闭环。
    *   算法遍历所有 Link，筛选出标记为 `is_overlap` 的连线。
    *   使用 `networkx` 寻找这些连线构成的最小环，定义为 **Overlap** 对象。
    *   **Overlap** 代表了主路径的一个物理交叉点，包含其中心坐标 (`center`)。

3.  **识别孔洞 (`_set_hole_fragments`)**:
    *   排除掉交叉区域的 Link，剩余的 Link 构成的闭环即为丝带之间的“孔洞”。
    *   **智能缝合 (`_close_open_holes`)**: 对于非闭合路径，丝带末端是开放的。算法使用 `KDTree` 寻找最近的邻居端点，尝试将开放的端点虚拟连接起来，形成封闭的孔洞区域，以便正确填充颜色。

### 4. 编织逻辑 (Weaving Logic / Z-Index)

这是算法的核心，用于决定在每个交叉点谁上谁下 (`_set_z_index`)。

*   **交替规则**: 沿着一条丝带行进，遇到交叉点时，其状态应遵循“上 -> 下 -> 上 -> 下”的交替规律。
*   **状态传播**:
    1.  遍历所有偏移路径。
    2.  对于每条路径，维护一个计数器 (`ind`)。
    3.  当遇到一个 `Overlap` 时，根据计数器的奇偶性 (`ind % 2`) 决定当前 Link 在该交叉点的 Z-Index (0 或 1)。
    4.  **Twin 同步**: 丝带的两侧偏移线必须保持一致的 Z-Index。
    5.  **冲突处理**: 算法通过 `visited` 标记避免重复处理，确保图遍历的一致性。

### 5. 几何重建 (Geometry Reconstruction)

根据计算出的拓扑信息，生成最终的可视化图形。

1.  **生成孔洞 (`HoleFragment`)**:
    *   直接使用第 3 步识别出的孔洞闭环坐标生成多边形。
    *   **分组着色 (`fragments_group`)**: 根据孔洞质心到全局中心的距离进行聚类分组。这使得对称图形（如曼陀罗、网格）的孔洞可以根据径向距离自动获得层次化的颜色。

2.  **生成丝带 (`RibbonFragment`)**:
    *   算法寻找 Z-Index 为 0 (Under/Bottom) 的路径段。
    *   它将“非交叉区域的线段”和“位于下层的交叉区域线段”连接起来，形成闭合的多边形。
    *   这些多边形即为编织结构中“被压在下面”或“浮在表面”的丝带片段。

3.  **调试可视化 (`_generate_debug_visuals`)**:
    *   如果开启 `DEV_MODE`，算法会额外绘制：
        *   红色半透明连线：主路径轨迹。
        *   黑色小点：原始路径顶点。
        *   红/蓝小点：计算出的交叉点（红色代表上层 Over，蓝色代表下层 Under）。

## 关键数据结构

*   **`Anchor`**: 几何点（顶点或交点）。
*   **`Link`**: 连接两个 Anchor 的有向线段，包含 `z_index` 信息。
*   **`Overlap`**: 描述一个交叉区域，管理 4 条 Link，是编织关系的容器。
*   **`Polyline` / `ParallelPolyline`**: 路径及其偏移线的几何抽象。
*   **`Lace`**: Manim `VGroup` 的子类，对外接口，负责协调上述所有步骤。

## 使用方法

```python
from manim_lace import Lace, StarRosette
from manim import *

class MyScene(Scene):
    def construct(self):
        # 1. 创建基础形状
        star = StarRosette(10, 3) 
        
        # 2. 应用 Lace 算法
        lace = Lace(
            star, 
            offset=0.15,          # 丝带宽度的一半
            hole_opacity=0.8,     # 孔洞不透明度
            stroke_color=WHITE,   # 丝带边缘颜色
            fill_color=BLUE       # 丝带填充颜色
        )
        
        self.add(lace)
```