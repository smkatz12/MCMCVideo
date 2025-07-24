from manimlib import *
import numpy as np
from manimlib.mobject.geometry import Rectangle
from manimlib.scene.interactive_scene import InteractiveScene

TBLUE = "#80B9FF"  # Deep Sky Blue
TPURPLE = "#C3B8FF"
TGREEN = "#80CFB9"
TPINK = "#FFA4E7"
TRED = "#FAB0AE"

# ── change this flag to switch pdf ───────────────────────────
NORMAL = False               # True → standard normal

class InteractiveProbabilitySamplingDemo(InteractiveScene):
    def construct(self):
        ## 1. axes & styling  --------------------------------------------------
        ax = Axes(
            x_range=[-4, 4],
            y_range=[-0.1, 0.6],
            width=10,
            height=6,
            x_axis_config=dict(
                stroke_color=WHITE,
                include_ticks=False,
                include_tip=False,
            ),
            y_axis_config=dict(
                stroke_opacity=0.0,
                include_ticks=False,
                include_tip=False,
            ),
        )
        ax.to_edge(DOWN)
        self.add(ax)
        self.ax = ax  # Store reference for mouse interactions

        ## 2. PDF curve --------------------------------------------------------
        pdf_curve = ax.get_graph(
            lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) if NORMAL else
            0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x)),
            color=WHITE
        )
    
        self.play(ShowCreation(pdf_curve), run_time=2)

        # ## 3. containers for scatter & histogram ------------------------------
        self.scatter_dots = VGroup()
        self.histogram_bars = VGroup()
        self.add(self.scatter_dots, self.histogram_bars)

        # ## 4. Initialize histogram data --------------------------------------
        self.bin_edges = np.linspace(-4, 4, 21)          # 20 equal-width bins
        self.counts = np.zeros(len(self.bin_edges) - 1)   # histogram counts
        
        # Add instruction text
        # instruction_text = Text(
        #     "Click near the x-axis to add sample points!",
        #     font_size=24,
        #     color=WHITE
        # )
        # instruction_text.to_edge(UP)
        # self.add(instruction_text)
        
    def on_mouse_press(self, point, button, mods):
        """Handle mouse clicks to add sample points"""
        super().on_mouse_press(point, button, mods)
        
        # Convert screen coordinates to axes coordinates
        axes_point = self.ax.p2c(point)
        x_coord = axes_point[0]
        
        # Only add points if click is within x-range and near the x-axis
        if -4 <= x_coord <= 4 and abs(axes_point[1]) < 0.15:
            self.add_sample_point(x_coord)
    
    def add_sample_point(self, x):
        """Add a sample point and update the histogram"""
        # Add scatter point
        dot = Dot(self.ax.c2p(x, 0), radius=0.15, fill_color=TBLUE, fill_opacity=0.6)
        self.scatter_dots.add(dot)

        # Update histogram data
        idx = np.searchsorted(self.bin_edges, x, side="right") - 1
        if 0 <= idx < len(self.counts):  # Make sure index is valid
            self.counts[idx] += 1

            # Rebuild histogram bars
            new_bars = VGroup()
            for i, c in enumerate(self.counts):
                if c == 0:
                    continue
                x_left = self.bin_edges[i]
                x_right = self.bin_edges[i+1]
                # Normalize the bar heights so that total area is 1
                bar_width = x_right - x_left
                bar_height = c / (bar_width * sum(self.counts))

                # Create rectangle for histogram bar
                bar = Rectangle(
                    width=bar_width * self.ax.x_axis.get_unit_size(),
                    height=bar_height * self.ax.y_axis.get_unit_size(),
                    fill_color=WHITE,
                    fill_opacity=0.3,
                    stroke_width=1,
                    stroke_color=WHITE
                )
                # Position the bar correctly
                bar.move_to(self.ax.c2p((x_left + x_right) / 2, bar_height / 2))
                new_bars.add(bar)

            # Animate: add dot, then update histogram
            self.play(FadeIn(dot), Transform(self.histogram_bars, new_bars), run_time=0.02)