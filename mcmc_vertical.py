from manimlib import *
import numpy as np
from manimlib.mobject.geometry import Rectangle

TBLUE = "#80B9FF"  # Deep Sky Blue
TPURPLE = "#C3B8FF"
TGREEN = "#80CFB9"
TPINK = "#FFA4E7"
TRED = "#FAB0AE"

# ── change this flag to switch pdf ───────────────────────────
NORMAL = False               # True → standard normal

class MCMC(Scene):
    def construct(self):
        ## 1. axes & styling  --------------------------------------------------
        ax = Axes(
            x_range=[-4, 4],
            y_range=[-0.1, 0.6],
            width=10,
            height=5.75,
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
        
        # Create a group to hold everything that will be rotated
        self.plot_group = VGroup()
        self.plot_group.add(ax)
        self.add(self.plot_group)

        ## 2. PDF curve --------------------------------------------------------
        pdf_curve = ax.get_graph(
            lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) if NORMAL else
            0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x)),
            color=WHITE
        )
        self.plot_group.add(pdf_curve)
    
        self.play(ShowCreation(pdf_curve), run_time=2)

        # ## 3. containers for scatter & histogram ------------------------------
        scatter_dots      = VGroup()
        histogram_bars    = VGroup()
        self.plot_group.add(scatter_dots, histogram_bars)

        # ## 4. choose sample points (here: scripted demo) -----------------------
        def mcmc_sampling(target_pdf, nsamples=500, xinit=0.0):
            x = xinit
            samples = [x]
            
            for i in range(nsamples):
                x_new = x + np.random.normal(0, 1)
                pdf_current = target_pdf(x)
                pdf_new = target_pdf(x_new)
                if pdf_current > 0:
                    acceptance_prob = min(1.0, pdf_new / pdf_current)
                else:
                    acceptance_prob = 1.0 if pdf_new > 0 else 0.0
                if np.random.random() < acceptance_prob:
                    x = x_new
                samples.append(x)
            return samples
        
        np.random.seed(4)
        sample_xs = mcmc_sampling(
            lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) if NORMAL else
            0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x)),
            nsamples=500,
            xinit=0.0
        )

        bin_edges = np.linspace(-4, 4, 21)          # 20 equal-width bins
        counts    = np.zeros(len(bin_edges) - 1)    # histogram counts

        # Add "Markov Chain Monte Carlo" text that writes on during sampling
        markov_chain_text = Text("Markov Chain", font="Gill Sans", font_size=48)
        monte_carlo_text = Text("Monte Carlo", font="Gill Sans", font_size=48)
        mcmc_text = VGroup(markov_chain_text, monte_carlo_text).arrange(RIGHT, buff=0.3)
        mcmc_text.to_edge(UP, buff=0.5)  # Center at top of screen
        
        # ## 5. animate each sample ---------------------------------------------
        sample_dots = []  # Store dots in order for later chain creation
        
        # Start writing the text as we begin sampling
        self.play(Write(mcmc_text), run_time=1.5)
        
        for i, x in enumerate(sample_xs):
            # scatter point
            dot = Dot(ax.c2p(x, 0), radius=0.15, fill_color=TBLUE, fill_opacity=0.6)
            scatter_dots.add(dot)
            sample_dots.append(dot)  # Store for chain creation

            # update histogram data
            idx = np.searchsorted(bin_edges, x, side="right") - 1
            counts[idx] += 1

            # rebuild histogram bars (delete & redraw)
            new_bars = VGroup()
            for j, c in enumerate(counts):
                if c == 0:
                    continue
                x_left  = bin_edges[j]
                x_right = bin_edges[j+1]
                # Normalize the bar heights so that total area is 1
                bar_width = x_right - x_left
                bar_height = c / (bar_width * sum(counts))

                # Create rectangle for histogram bar
                bar_width = x_right - x_left
                bar = Rectangle(
                    width=bar_width * ax.x_axis.get_unit_size(),
                    height=bar_height * ax.y_axis.get_unit_size(),
                    fill_color=WHITE,
                    fill_opacity=0.3,
                    stroke_width=1,
                    stroke_color=WHITE
                )
                # Position the bar correctly
                bar.move_to(ax.c2p((x_left + x_right) / 2, bar_height / 2))
                new_bars.add(bar)

            if i % 5 == 0 or i < 100:
                # Determine animation speed based on sample number
                if i < 10:  # First 10 samples: slow and clear
                    run_time = 0.05
                    highlight_time = 0.2
                elif i < 50:  # Next 40 samples: medium speed
                    run_time = 0.01
                    highlight_time = 0.05
                else:  # Remaining samples: very fast
                    run_time = 0.001
                    highlight_time = 0.001

                # animate: drop dot, then transform histogram
                highlight_dot = dot.copy().set_fill(TBLUE, opacity=0.8).scale(1.5)
                self.play(FadeIn(dot), Transform(histogram_bars, new_bars), FadeIn(highlight_dot, scale=0.5), run_time=run_time)
                
                self.play(
                    FadeOut(highlight_dot, scale=1.5),
                    run_time=highlight_time
                    )
            else:
                self.add(dot)

        # Store sample dots for chain creation
        self.sample_dots = sample_dots

        # ## 6. Rotate and position the entire plot -----------------------------
        # Rotate 90 degrees clockwise and move to left side
        self.play(
            self.plot_group.animate.scale(0.8).rotate(-PI/2).scale(0.7).to_edge(LEFT, buff=1),
            run_time=2
        )
        
        # # Then fade the histogram and PDF curve to focus on chain visualization
        # self.play(
        #     histogram_bars.animate.set_fill(opacity=0.1).set_stroke(opacity=0.2),
        #     pdf_curve.animate.set_stroke(opacity=0.3),
        #     run_time=0.5
        # )

        self.wait(1)
        
        # ## 7. Create MCMC chain visualization --------------------------------
        # Create a "slinky" effect where the chain connects dots in MCMC order
        # and gets progressively pulled away from the axis
        
        # After rotation, the "up" direction from the axis is now RIGHT
        chain_direction = RIGHT
        max_extension = 4.0  # Maximum distance the chain extends from the axis
        
        # Create the initial chain connecting all dots in MCMC order
        def get_chain_points(pull_factor=0.0):
            """Get chain points with varying extension based on pull_factor (0 to 1)"""
            chain_points = []
            for i, dot in enumerate(self.sample_dots):
                # Start from the dot's current position on the axis
                axis_point = dot.get_center()
                
                # Calculate how far this point extends based on its position in the chain
                # Earlier samples (lower i) extend further as we "pull" the chain
                # Keep the final point (last sample) on the axis
                if i == len(self.sample_dots) - 1:
                    extension = 0.0  # Final point stays on axis
                else:
                    extension_factor = (len(self.sample_dots) - i - 1) / (len(self.sample_dots) - 1)
                    extension = pull_factor * max_extension * extension_factor
                
                chain_point = axis_point + chain_direction * extension
                chain_points.append(chain_point)
            
            return chain_points
        
        # Create the chain path
        chain_path = VMobject()
        initial_points = get_chain_points(0.0)  # Start collapsed on axis
        chain_path.set_points_as_corners(initial_points)  # Use straight lines instead of smooth
        chain_path.set_stroke(TBLUE, width=3, opacity=0.8)
        
        # Add the chain immediately
        self.add(chain_path)
        
        # Center position for "Markov Chain" when it's alone
        center_position = UP * 3.5
        
        # Animate the "slinky" pulling effect with text sliding
        def update_chain(mob, alpha):
            """Update function for the chain pulling animation"""
            new_points = get_chain_points(alpha)
            mob.set_points_as_corners(new_points)  # Use straight lines
            
            # Also move the dots to follow the chain
            for i, dot in enumerate(self.sample_dots):
                dot.move_to(new_points[i])
        
        # Animate the slinky effect with text transformation
        self.play(
            UpdateFromAlphaFunc(chain_path, update_chain),
            FadeOut(monte_carlo_text),  # Fade out "Monte Carlo"
            markov_chain_text.animate.move_to(center_position),  # Slide "Markov Chain" to center
            # histogram_bars.animate.set_fill(opacity=0.1).set_stroke(opacity=0.2),
            histogram_bars.animate.set_fill(opacity=0.0).set_stroke(opacity=0.0),
            # pdf_curve.animate.set_stroke(opacity=0.3),
            pdf_curve.animate.set_stroke(opacity=0.0),
            run_time=2.5,
            rate_func=smooth
        )
        
        self.wait(1)

        # ## 8. Sequential highlight of first ~43 chain points ---------------
        # First fade the entire chain and dots to low opacity
        num_highlights = min(45, len(self.sample_dots))
        
        # Fade the chain and all dots to low opacity
        fade_animations = [chain_path.animate.set_stroke(opacity=0.3)]
        for dot in self.sample_dots:
            fade_animations.append(dot.animate.set_fill(opacity=0.3))
        
        self.play(*fade_animations, run_time=1.0)
        
        # Create a new chain path that we'll trace from the beginning
        trace_chain = VMobject()
        trace_chain.set_stroke(TBLUE, width=3, opacity=0.8)
        
        # Get the points for just the first num_highlights dots in reverse order (MCMC order)
        highlight_dots = [self.sample_dots[len(self.sample_dots) - i - 1] for i in range(num_highlights)]
        highlight_points = [dot.get_center() for dot in highlight_dots]
        
        # Animate tracing the chain and highlighting dots in sequence
        for i in range(num_highlights):
            # Highlight the current dot
            current_dot = highlight_dots[i]
            
            if i == 0:
                # First point - just create the initial point and highlight it
                trace_chain.set_points_as_corners(highlight_points[:1])
                self.add(trace_chain)
                self.play(
                    current_dot.animate.set_fill(TGREEN, opacity=1).scale(1.3),
                    run_time=0.1
                )
            else:
                # Scale and color the previous dot green
                prev_dot = highlight_dots[i-1]
                
                # Create a green line from previous to current dot
                green_line = Line(prev_dot.get_center(), current_dot.get_center())
                green_line.set_stroke(TGREEN, width=4, opacity=0.8)
                
                # Add the new segment to the traced chain
                new_points = highlight_points[:i+1]
                trace_chain.set_points_as_corners(new_points)
                
                # Fade the dot before the previous one (if it exists)
                if i >= 2:
                    dot_before_prev = highlight_dots[i-2]
                    dot_before_prev.set_fill(TBLUE, opacity=0.6).scale(1/1.2)
                
                self.play(
                    prev_dot.animate.set_fill(TGREEN, opacity=1).scale(1.2),
                    ShowCreation(green_line),
                    current_dot.animate.set_fill(TGREEN, opacity=1).scale(1.3),
                    run_time=0.15
                )
                
                # Remove the green line as it's now part of the main trace
                self.remove(green_line)
            
            # Only scale down the current dot if it's not the last one
            if i < num_highlights - 1:
                # Don't change color here - let it stay green until it becomes the "previous" dot
                self.play(
                    current_dot.animate.scale(1/1.3),
                    run_time=0.05
                )
                
        
        self.wait(0.5)
        
        # ## 9. Restore chain and dots to original appearance ---------------
        # Bring back the original chain opacity and reset all dot scaling/coloring
        restore_animations = [chain_path.animate.set_stroke(opacity=0.8)]
        
        # Reset all highlighted dots to uniform appearance
        for dot in highlight_dots:
            # Calculate the current scale factor and reset to original size
            current_height = dot.get_height()
            original_height = 0.3 * 0.8  # Original dot radius * 2
            scale_factor = original_height / current_height if current_height > 0 else 1.0
            restore_animations.append(dot.animate.set_fill(TBLUE, opacity=0.6).scale(scale_factor))
        
        # Also reset any other dots that might have been affected
        for dot in self.sample_dots:
            if dot not in highlight_dots:
                restore_animations.append(dot.animate.set_fill(TBLUE, opacity=0.6))
        
        # Remove the trace chain since we're going back to the original
        self.remove(trace_chain)
        
        self.play(*restore_animations, run_time=1.0)
        
        self.wait(1)
        
        # ## 10. Collapse the chain back down -----------------------------------
        def update_chain_collapse(mob, alpha):
            """Update function for collapsing the chain back to the axis"""
            # alpha goes from 0 to 1, but we want pull_factor to go from 1 to 0
            pull_factor = 1.0 - alpha
            new_points = get_chain_points(pull_factor)
            mob.set_points_as_corners(new_points)
            # Don't move the dots during collapse - they should return to axis positions
        
        # First move dots back to their axis positions
        axis_animations = []
        for i, dot in enumerate(self.sample_dots):
            axis_point = dot.get_center().copy()
            # Remove any extension to get back to axis
            if i < len(self.sample_dots) - 1:  # All except the last dot
                axis_point[0] = self.sample_dots[-1].get_center()[0]  # Use last dot's x position (on axis)
            axis_animations.append(dot.animate.move_to(axis_point))
        
        # Animate dots returning to axis and chain collapsing
        self.play(
            *axis_animations,
            UpdateFromAlphaFunc(chain_path, update_chain_collapse),
            FadeOut(markov_chain_text),  # Fade out "Markov Chain"
            histogram_bars.animate.set_fill(opacity=0.3).set_stroke(opacity=1),
            pdf_curve.animate.set_stroke(opacity=1),
            run_time=2.0,
            rate_func=smooth
        )
        
        self.wait(3)