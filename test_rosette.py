from manim_lace import StarRosette
from manim import *


class TestStarRosetteScene(Scene):
    def construct(self):
        # 标题
        title = Text("StarRosette Algorithm Test", font_size=36)
        title.to_edge(UP)
        self.add(title)

        star_type = "CA"

        # === 测试用例 1：简单的参数 ===
        n1, l1 = 7, 0
        rosette1 = StarRosette(n1, l1, star_type, stroke_color=TEAL, fill_color=BLUE, fill_opacity=0.5)
        # print(len(rosette1.get_anchors()))
        rosette1.shift(LEFT * 5)

        label1 = Text(f"n={n1}, level={l1}", font_size=24).next_to(rosette1, DOWN)

        # === 测试用例 2：复杂的参数 ===
        n2, l2 = 10, 1
        rosette2 = StarRosette(n2, l2, star_type, stroke_color=PINK, fill_color=LIGHT_PINK, fill_opacity=0.5)
        rosette2.shift(LEFT * 1).scale(0.6)

        label2 = Text(f"n={n2}, level={l2}", font_size=24).next_to(rosette2, DOWN)

        self.add(label1, label2)

        # === 测试用例 3：复杂的参数 ===
        n3, l3 = 10, 3
        rosette3 = StarRosette(n3, l3, star_type, stroke_color=ORANGE, fill_color=YELLOW, fill_opacity=0.5)

        rosette3.scale(0.6).shift(RIGHT * 3.5)

        label3 = Text(f"n={n3}, level={l3}", font_size=24).next_to(rosette3, DOWN)

        self.add(label1, label2, label3)

        self.play(Create(rosette1), Create(rosette2), Create(rosette3), run_time=5, rate_func=linear)

        self.wait(1)


with tempconfig({"quality": "medium_quality", "preview": True, "verbosity": "WARNING"}):
    TestStarRosetteScene().render()
