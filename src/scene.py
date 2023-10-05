from manim import *


# class LSTMAnimation(Scene):
#     def construct(self):
#         title = Text("LSTM Operation")
#         description = Text("Long Short-Term Memory").next_to(title, DOWN)
#
#         self.play(Write(title))
#         self.play(FadeIn(description))
#         self.wait(1)
#         self.play(FadeOut(title), FadeOut(description))
#
#         # Add more visualizations and animations related to LSTM as needed
#


# class RNNUnroll(Scene):
#     def construct(self):
#         title = Text("RNN Unrolling")
#         self.play(Write(title))
#         self.wait(1)
#
#         # Create circles to represent the RNN nodes
#         nodes = [Circle(radius=0.5, fill_opacity=0.8, fill_color=BLUE) for _ in range(4)]
#         for i in range(len(nodes)):
#             if i == 0:
#                 nodes[i].next_to(title, DOWN * 2)
#             else:
#                 nodes[i].next_to(nodes[i - 1], RIGHT * 2)
#
#         # Create texts inside the nodes
#         texts = [Text(f"t={i}") for i in range(len(nodes))]
#         for i in range(len(nodes)):
#             texts[i].move_to(nodes[i].get_center())
#
#         # Add arrows
#         arrows = [Arrow(start=node1.get_right(), end=node2.get_left()) for node1, node2 in zip(nodes[:-1], nodes[1:])]
#
#         for node, text, arrow in zip(nodes, texts, arrows):
#             self.play(FadeIn(node), Write(text))
#             if arrow != arrows[-1]:
#                 self.play(Create(arrow))
#
#         self.wait(1)
#         self.play(*[FadeOut(mobj) for mobj in self.mobjects])


class LSTMCell(Scene):
    def construct(self):
        title = Text("LSTM Cell")
        self.play(Write(title))
        self.wait(1)

        # Create the LSTM cell rectangle
        cell_rect = RoundedRectangle(width=5, height=3, fill_opacity=0.2, fill_color=BLUE)
        cell_rect.next_to(title, DOWN * 2)

        # Create the gate rectangles
        gate_names = ["Input Gate", "Forget Gate", "Output Gate"]
        gates = [Rectangle(width=1.5, height=1, fill_opacity=0.5, fill_color=GREEN) for _ in gate_names]
        for i, gate in enumerate(gates):
            if i == 0:
                gate.next_to(cell_rect, LEFT, aligned_edge=UP)
            else:
                gate.next_to(gates[i - 1], DOWN, buff=0.5)

        # Create the text for each gate
        gate_texts = [Text(name).scale(0.2).move_to(gate) for name, gate in zip(gate_names, gates)]

        # Display the cell and gates
        self.play(FadeIn(cell_rect))
        for gate, gate_text in zip(gates, gate_texts):
            self.play(FadeIn(gate), Write(gate_text))
            self.wait(1)

        self.wait(1)
        self.play(*[FadeOut(mobj) for mobj in self.mobjects])
