# manim_lace

## 项目概述

`manim_lace` 是一个基于 [Manim](https://www.manim.community/)（数学动画引擎）的 Python 项目。它提供了一套算法和类，旨在实现以下功能：
1.  **生成"星形玫瑰"图案**：基于核算法（kernel algorithms）生成几何星形/花朵形状。
2.  **创建"Lace"（交错编织）效果**：将任意 Manim 路径（如玫瑰线或网格）转换为具有交错效果的丝带图案，自动处理上下遮挡关系以模拟编织视觉效果。

## 核心组件

### 1. `manim_lace` 包

*   **`Lace` (`manim_lace/lace.py`)**：核心类。它接收一个 `VMobject`（路径），计算交点，并生成一个表示编织丝带的新 `VGroup`。
    *   **特性**：自动检测交点，根据丝带宽度计算偏移量，以及自动着色。
*   **`StarRosette` (`manim_lace/rosette.py`)**：`Polygram` 的子类，用于生成对称的星形图案。
    *   **参数**：`n`（点数），`level`（复杂度/步长），`star_type`（例如 'PA', 'PO', 'CA' 等）。

### 2. 测试与可视化

*   **`test_lace.py`**：包含测试场景（`TestLaceScene0`, `TestLaceScene1` 等），用于渲染应用于 `StarRosette` 和网格的 `Lace` 对象。
*   **`test_rosette.py`**：通常包含用于验证 `StarRosette` 对象几何形状的测试场景。

## 开发环境设置

### 依赖管理

本项目使用 `uv` 进行依赖管理，配置位于 `pyproject.toml` 文件中。

### 常用命令

*   **安装依赖**：
    ```bash
    uv sync
    ```

*   **运行可视化测试**：
    测试文件是直接渲染 Manim 场景的脚本。
    ```bash
    python test_lace.py
    python test_rosette.py
    ```
    *注意：这些脚本通常使用 `tempconfig` 来立即渲染输出（例如输出到 `media/` 目录）。*

*   **渲染特定场景 (Manim CLI)**：
    你也可以使用标准的 Manim CLI 从文件中渲染场景。
    ```bash
    # 示例：渲染文件中的特定场景
    manim -pql test_lace.py TestLaceScene0
    ```

## 目录结构

*   `manim_lace/`: 软件包源代码。
    *   `lace.py`: 编织交错逻辑。
    *   `rosette.py`: 星形生成逻辑。
    *   `swatches.py`: 调色板工具。
*   `test_*.py`: 用于测试和可视化库功能的脚本。
*   `media/`: Manim 渲染的默认输出目录。
*   `CLAUDE.md`: 原始项目上下文和架构说明（参考）。
