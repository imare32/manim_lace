from manim_lace import Lace, StarRosette
from manim import *


class TestLaceScene0(Scene):
    def construct(self):
        star = StarRosette(10, 2).scale(1)
        lace = Lace(star, offset=0.1, stroke_color=BLACK, stroke_width=1)
        self.add(lace)


class TestLaceScene1(Scene):
    def construct(self):
        star = StarRosette(10, 3).scale(1)
        lace = Lace(star, offset=0.1, stroke_color=BLACK, stroke_width=1)
        self.add(lace)


class TestLaceScene2(Scene):
    def construct(self):
        grid = NumberPlane().background_lines
        # grid = NumberPlane(x_range=(-2, 2), y_range=(-2, 2)).background_lines
        # grid = VGroup(Line(LEFT * 5, RIGHT * 5), Line(DOWN * 5, UP * 5))
        lace = Lace(grid, offset=0.1, stroke_color=BLACK, stroke_width=1, hole_opacity=0.5, hole_group_method="by_distance")
        self.add(lace)


class TestLaceScene3(Scene):
    def construct(self):
        obj = VGroup(Rectangle(width=5, height=5), Star(outer_radius=4))
        lace = Lace(obj, offset=0.1, stroke_color=BLACK, stroke_width=1)
        self.add(lace)


class TestLaceScene4(Scene):
    def func(self, t):
        return np.sin(2 * t), np.sin(3 * t), 0

    def construct(self):
        func = ParametricFunction(self.func, t_range=(0, TAU, np.pi / 20), use_smoothing=False).scale(3)
        func.shift(DOWN * 0.5)
        func.set_color(BLACK)
        func.set_stroke(width=1)
        lace = Lace(func, offset=0.1, stroke_color=BLACK, stroke_width=1)
        self.add(
            lace,
            # func,
        )
        for p in func.get_anchors():
            self.add(Dot(p, color=RED))


if __name__ == "__main__":
    with tempconfig({"quality": "medium_quality", "preview": False, "verbosity": "WARNING", "background_color": WHITE}):
        TestLaceScene0().render()
    with tempconfig({"quality": "medium_quality", "preview": False, "verbosity": "WARNING", "background_color": WHITE}):
        TestLaceScene1().render()
    with tempconfig({"quality": "low_quality", "preview": False, "verbosity": "WARNING", "background_color": WHITE}):
        TestLaceScene2().render()
    with tempconfig({"quality": "low_quality", "preview": False, "verbosity": "WARNING", "background_color": WHITE}):
        TestLaceScene3().render()
    with tempconfig({"quality": "low_quality", "preview": False, "verbosity": "WARNING", "background_color": WHITE}):
        TestLaceScene4().render()
