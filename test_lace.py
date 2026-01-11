from manim_lace import Lace, StarRosette
from manim import *


class TestLaceScene0(Scene):
    def construct(self):
        star = StarRosette(10, 2).scale(1)
        lace = Lace(star, offset=0.1, stroke_color=BLACK, stroke_width=1, fill_color=RED, fill_opacity=0.5)
        self.add(lace)


class TestLaceScene1(Scene):
    def construct(self):
        star = StarRosette(10, 3).scale(1)
        lace = Lace(star, offset=0.1, stroke_color=BLACK, stroke_width=1, fill_color=RED, fill_opacity=0.5)
        self.add(lace)


class TestLaceScene2(Scene):
    def construct(self):
        grid = NumberPlane().background_lines
        lace = Lace(grid, offset=0.1, stroke_color=BLACK, stroke_width=1, fill_color=RED, fill_opacity=0.5)
        self.add(lace)


class TestLaceScene3(Scene):
    def construct(self):
        lace = Lace(
            VGroup(Rectangle(width=5, height=5), Star(outer_radius=4)),
            offset=0.1,
            stroke_color=BLACK,
            stroke_width=1,
            fill_color=RED,
            fill_opacity=0.5,
        ).shift(DOWN * 0.5)
        self.add(lace)


if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "preview": False, "verbosity": "WARNING", "background_color": WHITE}):
        TestLaceScene0().render()
    with tempconfig({"quality": "low_quality", "preview": False, "verbosity": "WARNING", "background_color": WHITE}):
        TestLaceScene1().render()
    with tempconfig({"quality": "low_quality", "preview": False, "verbosity": "WARNING", "background_color": WHITE}):
        TestLaceScene2().render()
    with tempconfig({"quality": "low_quality", "preview": False, "verbosity": "WARNING", "background_color": WHITE}):
        TestLaceScene3().render()
