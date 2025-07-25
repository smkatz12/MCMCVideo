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
        # region 1. axes & styling  --------------------------------------------------
        ax = Axes(
            x_range=[-4, 4],
            y_range=[-0.03, 0.6],
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

        # endregion

        # region 2. PDF curve --------------------------------------------------------
        pdf_curve = ax.get_graph(
            lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) if NORMAL else
            0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x)),
            color=WHITE
        )
        self.plot_group.add(pdf_curve)
    
        self.play(ShowCreation(pdf_curve), run_time=2)

        # endregion

        # region 3. containers for scatter & histogram ------------------------------
        scatter_dots      = VGroup()
        histogram_bars    = VGroup()
        self.plot_group.add(scatter_dots, histogram_bars)

        # endregion

        # region 4. choose sample points (here: scripted demo) -----------------------
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

        # endregion

        # region 5. animate each sample ---------------------------------------------
        sample_dots = []  # Store dots in order for later chain creation
        
        # Start writing the text as we begin sampling
        self.play(Write(mcmc_text), run_time=1.5)
        
        for i, x in enumerate(sample_xs):
            # scatter point
            dot = Dot(ax.c2p(x, 0), radius=0.1, fill_color=TBLUE, fill_opacity=0.6)
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

        # endregion

        # region 6. Create MCMC chain visualization --------------------------------
        # Create a "slinky" effect where the chain connects dots in MCMC order
        # and gets progressively pulled away from the axis
        
        # Store original axis positions for each dot
        original_axis_positions = [dot.get_center().copy() for dot in self.sample_dots]
        
        # After rotation, the "up" direction from the axis is now RIGHT
        chain_direction = UP
        max_extension = 80.0  # Maximum distance the chain extends from the axis
        
        # Create the initial chain connecting all dots in MCMC order
        def get_chain_points(pull_factor=0.0):
            """Get chain points with varying extension based on pull_factor (0 to 1)"""
            chain_points = []
            for i, dot in enumerate(self.sample_dots):
                # Start from the original axis position, not current position
                axis_point = original_axis_positions[i]
                
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
        center_position = LEFT * 3.5 + UP * 0.5

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

        # endregion

        # region 7. Sequential highlight of first ~43 chain points ---------------
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

        # endregion
    
        # region 8. Collapse the chain back down -----------------------------------
        def update_chain_collapse(mob, alpha):
            """Update function for collapsing the chain back to the axis"""
            # alpha goes from 0 to 1, but we want pull_factor to go from 1 to 0
            pull_factor = 1.0 - alpha
            new_points = get_chain_points(pull_factor)
            mob.set_points_as_corners(new_points)
            
            # Move the dots to follow the collapsing chain
            for i, dot in enumerate(self.sample_dots):
                dot.move_to(new_points[i])
        
        # First reset all highlighted dots back to normal appearance but keep them in extended positions
        reset_animations = []
        for dot in highlight_dots:
            # Reset any scaling and color back to normal but don't move them yet
            current_height = dot.get_height()
            original_height = 0.2  # Original dot radius * 2 (radius was 0.1)
            scale_factor = original_height / current_height if current_height > 0 else 1.0
            reset_animations.append(dot.animate.set_fill(TBLUE, opacity=0.6).scale(scale_factor))
        
        # Also reset any other dots that might have been affected
        for dot in self.sample_dots:
            if dot not in highlight_dots:
                reset_animations.append(dot.animate.set_fill(TBLUE, opacity=0.6))
        
        # Make sure we're using the right chain for collapse - remove trace_chain and use original
        self.remove(trace_chain)
        
        # Reset the original chain to full extension and move dots to match
        full_extension_points = get_chain_points(1.0)
        chain_path.set_points_as_corners(full_extension_points)
        chain_path.set_stroke(opacity=0.8)
        
        # Move all dots to their full extension positions
        for i, dot in enumerate(self.sample_dots):
            dot.move_to(full_extension_points[i])
        
        # Play the reset animations (appearance only, no movement since dots are already positioned)
        if reset_animations:
            self.play(*reset_animations, run_time=0.5)
        
        # Animate chain collapsing with dots following along
        self.play(
            UpdateFromAlphaFunc(chain_path, update_chain_collapse),
            FadeOut(markov_chain_text),  # Fade out "Markov Chain"
            histogram_bars.animate.set_fill(opacity=0.3).set_stroke(opacity=1),
            pdf_curve.animate.set_stroke(opacity=1),
            run_time=2.0,
            rate_func=smooth
        )
        
        self.wait(3)

        # endregion

        # region 9. Clean up -----------------------------------
        # Fade out all dots and the chain and the histogram and the pdf curve
        fade_animations = [chain_path.animate.set_stroke(opacity=0.0)]
        for dot in self.sample_dots:
            fade_animations.append(dot.animate.set_fill(opacity=0.0))
        self.play(*fade_animations, 
                  histogram_bars.animate.set_fill(opacity=0.0).set_stroke(opacity=0.0),
                  pdf_curve.animate.set_stroke(opacity=0.0),run_time=1.0)
        
        # endregion

        # region 10. Show the first iteration of chain creation -----------------------------------
        # Show the first dot in the chain
        first_dot = self.sample_dots[-1]
        self.play(first_dot.animate.set_fill(TBLUE, opacity=0.8), run_time=1)

        current_x = sample_xs[-1]  # First sample position (current state)
        
        # Add label for the blue point (current state x)
        x_label = Tex(r"x", font_size=36)
        x_label.set_color(TBLUE)
        x_label.next_to(first_dot, DOWN, buff=0.15)
        x_label.align_to(ax.c2p(0, -0.05), DOWN)  # Align to consistent baseline
        self.play(Write(x_label), run_time=0.5)

        # Show the creation of the Gaussian kernel with label as Kernel
        gaussian_kernel = ax.get_graph(
            lambda x: np.exp(-(x-current_x)**2 / 2) / np.sqrt(2 * np.pi),
            color=TGREEN,
            x_range=[current_x-3, current_x+3]
        )
        gaussian_kernel_label = Text("Kernel", font="Gill Sans", font_size=36)
        gaussian_kernel_label.set_color(TGREEN)
        gaussian_kernel_label.next_to(gaussian_kernel, UP, buff=0.2)

        self.play(
            ShowCreation(gaussian_kernel),
            Write(gaussian_kernel_label),
            run_time=2)
        
        self.wait(1)

        # endregion

        # region 11. Animate sampling from the kernel with density-based movement --------
        target_x = sample_xs[-2]   # Second sample position (target from chain)
        
        # Create the sampling dot that will move along the kernel
        sampling_dot = Dot(radius=0.1, fill_color=TGREEN, fill_opacity=1.0)
        sampling_dot.move_to(ax.c2p(current_x - 3.0, 0))  # Start at current_x - 3
        
        # Show the sampling dot appearing at current_x - 3
        self.play(FadeIn(sampling_dot, scale=0.5), run_time=0.05)
        
        # Define the kernel function for speed calculation
        def kernel_pdf(x):
            return np.exp(-(x - current_x)**2 / 2) / np.sqrt(2 * np.pi)
        
        # Create the movement animation with density-based speed
        def update_sampling_dot(mob, alpha):
            # Movement parameters
            oscillation_range = 3.0  # Oscillate ±3 around current_x
            total_oscillations = 2.5  # 2 full oscillations + 0.5 to stop at target
            
            # Calculate which phase we're in
            phase = alpha * total_oscillations
            
            if phase < 2.0:  # First two complete oscillations
                # Complete oscillations between current_x-3 and current_x+3
                # Use cosine starting at -1 (which gives us current_x-3) for continuity
                oscillation_alpha = phase / 2.0  # Maps to 0-1 for two oscillations
                # Start at current_x-3 (cos(0) = 1, so we use -cos to start at -1)
                x_position = current_x - oscillation_range * np.cos(oscillation_alpha * 2 * PI)
                
            else:  # Third partial oscillation - stop at target
                # Start from current_x-3 and move towards target_x
                remaining_phase = phase - 2.0  # 0 to 0.5
                progress_to_target = remaining_phase / 0.5  # Maps to 0-1
                
                # Start at current_x-3 and interpolate to target_x
                start_x = current_x - oscillation_range
                x_position = start_x + (target_x - start_x) * progress_to_target
            
            # Apply density-based speed modulation
            density = kernel_pdf(x_position)
            max_density = kernel_pdf(current_x)
            
            # Create a speed factor (higher density = slower movement)
            if max_density > 0:
                density_factor = density / max_density
                # Slow down in high-density regions (reduced intensity)
                speed_multiplier = 0.6 + 0.4 * (1 - density_factor)  # Speed varies from 0.6 to 1.0
            else:
                speed_multiplier = 1.0
            
            # Apply consistent speed modulation for all phases
            effective_x = x_position * speed_multiplier + x_position * (1 - speed_multiplier) * 0.3
            
            mob.move_to(ax.c2p(effective_x, 0))
        
        # Animate the density-based sampling movement
        self.play(
            UpdateFromAlphaFunc(sampling_dot, update_sampling_dot),
            run_time=3.0,  # Faster duration while still showing the oscillations clearly
            rate_func=smooth
        )

        # Add label for the green point (proposed state x') after it reaches final position
        x_prime_label = Tex(r"x'", font_size=36)
        x_prime_label.set_color(TGREEN)
        x_prime_label.next_to(sampling_dot, DOWN, buff=0.15)
        x_prime_label.align_to(ax.c2p(0, -0.05), DOWN)  # Align to same baseline as x label
        self.play(Write(x_prime_label), run_time=0.3)

        self.wait(0.5)

        self.play(
            FadeOut(gaussian_kernel, run_time=0.5),
            FadeOut(gaussian_kernel_label, run_time=0.5),
        )
        
        # endregion

        # region 12. Draw target density back on the screen -------------------
        pdf_curve = ax.get_graph(
            lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) if NORMAL else
            0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x)),
            color=WHITE
        )
        self.plot_group.add(pdf_curve)
    
        self.play(ShowCreation(pdf_curve), run_time=2)

        # endregion
        