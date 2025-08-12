from manimlib import *
import numpy as np
import random
from manimlib.mobject.geometry import Rectangle

TBLUE = "#80B9FF"  # Deep Sky Blue
TPURPLE = "#C3B8FF"
TGREEN = "#80CFB9"
TPINK = "#FFA4E7"
TRED = "#FAB0AE"
TYELLOW = "#FEEAAD"

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

        self.wait(1.0)

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

        # region 11. Animate sampling from the kernel with random position sampling --------
        target_x = sample_xs[-2]   # Second sample position (target from chain)
        
        # Create the sampling dot that will move along the kernel
        sampling_dot = Dot(radius=0.1, fill_color=TGREEN, fill_opacity=1.0)
        sampling_dot.move_to(ax.c2p(current_x - 3.0, 0))  # Start at current_x - 3
        
        # Show the sampling dot appearing at current_x - 3
        self.play(FadeIn(sampling_dot, scale=0.5), run_time=0.5)
        
        # Define the kernel function for sampling
        def kernel_pdf(x):
            return np.exp(-(x - current_x)**2 / 2) / np.sqrt(2 * np.pi)
        
        # Function to sample from the kernel using inverse transform sampling (approximate)
        def sample_from_kernel():
            # Use rejection sampling for the Gaussian kernel
            while True:
                # Sample uniformly from the range
                x_candidate = random.uniform(current_x - 3, current_x + 3)
                # Accept with probability proportional to kernel density
                acceptance_prob = kernel_pdf(x_candidate) / kernel_pdf(current_x)
                if random.random() < acceptance_prob:
                    return x_candidate
        
        # Set random seed for reproducibility
        random.seed(8)
        
        # Generate several random sample positions from the kernel
        random_sample_positions = []
        for _ in range(10):  # Generate 10 random samples from the kernel
            random_sample_positions.append(sample_from_kernel())
        random_sample_positions.append(target_x)  # End at the actual target value
        
        # Animate through the random positions sampled from the kernel
        for i, x_pos in enumerate(random_sample_positions):
            target_pos = ax.c2p(x_pos, 0)
            if i < len(random_sample_positions) - 1:
                # Quick movements for intermediate positions
                self.play(sampling_dot.animate.move_to(target_pos), run_time=0.15)
            else:
                # Slower final movement to target
                self.play(sampling_dot.animate.move_to(target_pos), run_time=0.4)

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

        # region 13. Draw lines from x-axis points to function values and add labels --------
        # Define the target PDF function for evaluation
        def target_pdf(x):
            if NORMAL:
                return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
            else:
                return 0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x))
        
        # Get the current positions and function values
        x_pos = current_x
        x_prime_pos = target_x
        f_x_value = target_pdf(x_pos)
        f_x_prime_value = target_pdf(x_prime_pos)
        
        # Create vertical lines from x-axis to function
        x_line = Line(
            ax.c2p(x_pos, 0),
            ax.c2p(x_pos, f_x_value),
            color=TBLUE,
            stroke_width=3
        )
        
        x_prime_line = Line(
            ax.c2p(x_prime_pos, 0),
            ax.c2p(x_prime_pos, f_x_prime_value),
            color=TGREEN,
            stroke_width=3
        )
        
        # Animate drawing both lines simultaneously
        self.play(
            ShowCreation(x_line),
            ShowCreation(x_prime_line),
            run_time=1.5
        )
        
        # Create function value labels
        f_x_label = Tex(f"f(x) = {f_x_value:.2f}", font_size=28)
        f_x_label.set_color(TBLUE)
        f_x_label.next_to(ax.c2p(x_pos, f_x_value), UP, buff=0.1)
        
        f_x_prime_label = Tex(f"f(x') = {f_x_prime_value:.2f}", font_size=28)
        f_x_prime_label.set_color(TGREEN)
        f_x_prime_label.next_to(ax.c2p(x_prime_pos, f_x_prime_value), UP, buff=0.1)
        
        # Create dots at the function value points
        f_x_dot = Dot(ax.c2p(x_pos, f_x_value), radius=0.08, fill_color=TBLUE, fill_opacity=1.0)
        f_x_prime_dot = Dot(ax.c2p(x_prime_pos, f_x_prime_value), radius=0.08, fill_color=TGREEN, fill_opacity=1.0)
        
        # Animate adding the labels and dots simultaneously
        self.play(
            Write(f_x_label),
            Write(f_x_prime_label),
            FadeIn(f_x_dot, scale=0.5),
            FadeIn(f_x_prime_dot, scale=0.5),
            pdf_curve.animate.set_stroke(opacity=0.3),  # Fade the function curve
            run_time=1.0
        )
        
        self.wait(1)

        # endregion

        # region 14. Show Chain Creation Rules --------
        # Create the rules box in the upper left corner
        rules_box = Rectangle(
            width=3.0,
            height=2.35,
            fill_color=BLACK,
            fill_opacity=0.8,
            stroke_color=WHITE,
            stroke_width=2
        )
        rules_box.to_edge(LEFT, buff=0.3).to_edge(UP, buff=0.3)
        
        # Create the title with slightly more buffer
        rules_title = Text("Chain Creation Rules", font="Gill Sans", font_size=24)
        rules_title.set_color(WHITE)
        rules_title.next_to(rules_box.get_top(), DOWN, buff=0.15)
        
        # Create rule 1 (currently applicable) with indented action
        rule1_condition = Tex(r"\text{If } f(x') > f(x):", font_size=20)
        rule1_condition.set_color(WHITE)
        
        rule1_action = Tex(r"\text{Accept } x'", font_size=20)
        # rule1_action.set_color(TGREEN)
        rule1_action.set_color(WHITE)
        
        # Arrange with left alignment and indent the action
        rule1 = VGroup(rule1_condition, rule1_action).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        rule1_action.shift(RIGHT * 0.3)  # Indent the action
        rule1.next_to(rules_title, DOWN, buff=0.3).align_to(rules_box.get_left(), LEFT).shift(RIGHT * 0.2)
        
        # Create placeholder for rule 2 (to be shown later) with indented action
        rule2_condition = Tex(r"\text{If } f(x') < f(x):", font_size=20)
        rule2_condition.set_color(WHITE)
        
        rule2_action = Tex(r"\text{Accept with prob. } \frac{f(x')}{f(x)}", font_size=18)
        # rule2_action.set_color(YELLOW)
        rule2_action.set_color(WHITE)
        
        # Arrange with left alignment and indent the action
        rule2 = VGroup(rule2_condition, rule2_action).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        rule2_action.shift(RIGHT * 0.3)  # Indent the action
        rule2.next_to(rule1, DOWN, buff=0.3).align_to(rule1, LEFT)
        
        # Group everything together (no need to center since we're using manual positioning)
        rules_content = VGroup(rules_title, rule1, rule2)
        
        # Show the box and title first
        self.play(
            FadeIn(rules_box),
            Write(rules_title),
            run_time=1.0
        )
        
        # Since f(x') > f(x) in our case, show rule 1
        if f_x_prime_value > f_x_value:
            # Highlight that this rule applies
            self.play(
                Write(rule1_condition),
                run_time=1.0
            )
            self.play(
                Write(rule1_action),
                run_time=1.0
            )
        
        self.wait(1)

        # endregion

        # region 15. Accept the proposal and clean up --------
        # Add "ACCEPT!" text next to the x' sample
        accept_text = Text("ACCEPT!", font="Gill Sans", font_size=32, color=TGREEN, weight=BOLD)
        accept_text.next_to(sampling_dot, UP, buff=0.3)
        
        # Remove the vertical lines and function dots
        self.play(
            Write(accept_text),
            FadeOut(x_line),
            FadeOut(x_prime_line),
            FadeOut(f_x_label),
            FadeOut(f_x_prime_label),
            FadeOut(f_x_dot),
            FadeOut(f_x_prime_dot),
            FadeOut(x_prime_label),
            pdf_curve.animate.set_stroke(opacity=0.0),
            run_time=1.0
        )
        
        # # Change the green sampling dot to blue since it's being accepted into the chain
        # self.play(
        #     sampling_dot.animate.set_fill(TBLUE, opacity=0.8),
        #     pdf_curve.animate.set_stroke(opacity=0.0),
        #     run_time=0.8
        # )
        
        self.wait(0.5)

        # endregion

        # region 16. Add new point to the chain --------
        # Calculate where the new dot should be positioned in the extended chain
        # This simulates the chain expansion that happened earlier
        # The new dot will be the second-to-last point (since the last point stays on axis)
        
        # Get the current chain configuration with the new point
        # We need to add the sampling_dot as the new second point in the chain
        extended_sample_dots = self.sample_dots + [sampling_dot]  # Add new dot to chain
        
        # Calculate the new chain positions (similar to get_chain_points but for the new configuration)
        def get_new_chain_points(pull_factor=1.0):
            """Get chain points including the new accepted sample"""
            chain_points = []
            total_dots = len(extended_sample_dots)
            
            for i, dot in enumerate(extended_sample_dots):
                if i < len(self.sample_dots):
                    # Use original positions for existing dots
                    axis_point = original_axis_positions[i]
                else:
                    # New dot starts from its current position on the axis
                    axis_point = dot.get_center()
                
                # Calculate extension - keep the final point (last sample) on the axis
                if i == total_dots - 1:
                    extension = 0.0  # Final point stays on axis
                else:
                    extension_factor = (total_dots - i - 1) / (total_dots - 1)
                    extension = pull_factor * max_extension * extension_factor
                
                chain_point = axis_point + chain_direction * extension
                chain_points.append(chain_point)
            
            return chain_points
        
        # Get the target position for the new dot
        new_chain_points = get_new_chain_points(1.0)
        new_dot_target = new_chain_points[-2]  # Second to last position (new dot)
        # The first_dot stays where it is (on the axis) as the last point in the chain
        
        # Keep the same x-position but move to the chain y-position
        current_position = sampling_dot.get_center()
        target_position = np.array([current_position[0], new_dot_target[1], current_position[2]])

        # Draw the connecting line between the two dots
        chain_line = Line(first_dot.get_center(), target_position)
        chain_line.set_stroke(TBLUE, width=3, opacity=0.8)
        
        # Move only the new dot to its chain position (same x, new y)
        self.play(
            sampling_dot.animate.move_to(target_position).set_fill(TBLUE, opacity=0.8),
            x_label.animate.next_to(target_position, DOWN, buff=0.25), 
            ShowCreation(chain_line),
            FadeOut(accept_text),
            run_time=1.5
        )
        
        self.wait(1)

        # endregion

        # region 17. Repeat the process for the next sample --------
        # Now we'll repeat the sampling process, starting from the current state (sampling_dot)
        # and moving to the next sample in the chain
        
        # Update the current and target positions for the next iteration
        prev_x = target_x  # The sampling_dot position becomes the new current state
        next_target_x = sample_xs[-3]  # Third sample position (next target from chain)
        
        # Get the position of the current chain point for the temporary x-axis
        current_chain_pos = sampling_dot.get_center()
        current_chain_y = current_chain_pos[1]
        
        # Create a temporary x-axis aligned with the current chain point
        # Use screen coordinates directly
        left_point = ax.c2p(-4, 0)
        right_point = ax.c2p(4, 0)
        left_point[1] = current_chain_y  # Set y to match chain point
        right_point[1] = current_chain_y  # Set y to match chain point
        
        temp_x_axis = Line(
            left_point,
            right_point,
            color=WHITE,
            stroke_width=2,
            stroke_opacity=0.3
        )
        
        # Show the temporary x-axis
        self.play(ShowCreation(temp_x_axis), run_time=1.0)
        
        # Fade the previous iteration elements to lower opacity
        self.play(
            first_dot.animate.set_fill(opacity=0.3),
            chain_line.animate.set_stroke(opacity=0.3),
            run_time=0.5
        )
        
        # Show the creation of a new Gaussian kernel centered at the new current state
        # But position it relative to the temporary axis
        new_gaussian_kernel = ax.get_graph(
            lambda x: np.exp(-(x-prev_x)**2 / 2) / np.sqrt(2 * np.pi),
            color=TGREEN,
            x_range=[prev_x-3, prev_x+3]
        )
        # Shift the kernel to align with the temporary axis
        kernel_shift = current_chain_y - ax.c2p(0, 0)[1]  # Calculate shift from main axis
        new_gaussian_kernel.shift(UP * kernel_shift)
        
        new_gaussian_kernel_label = Text("Kernel", font="Gill Sans", font_size=36)
        new_gaussian_kernel_label.set_color(TGREEN)
        new_gaussian_kernel_label.next_to(new_gaussian_kernel, UP, buff=0.2)

        self.play(
            ShowCreation(new_gaussian_kernel),
            Write(new_gaussian_kernel_label),
            run_time=2
        )
        
        self.wait(1)

        # Create a new sampling dot for the second iteration on the temporary axis
        new_sampling_dot = Dot(radius=0.1, fill_color=TGREEN, fill_opacity=1.0)
        # Position it on the temporary x-axis at prev_x - 3
        temp_axis_start_pos = ax.c2p(prev_x - 3.0, 0)
        temp_axis_start_pos[1] = current_chain_y  # Align with temporary axis
        new_sampling_dot.move_to(temp_axis_start_pos)
        
        # Show the new sampling dot appearing
        self.play(FadeIn(new_sampling_dot, scale=0.5), run_time=0.5)
        
        # Define the new kernel function for sampling
        def new_kernel_pdf(x):
            return np.exp(-(x - prev_x)**2 / 2) / np.sqrt(2 * np.pi)
        
        # Function to sample from the kernel using rejection sampling
        def sample_from_new_kernel():
            # Use rejection sampling for the Gaussian kernel
            while True:
                # Sample uniformly from the range
                x_candidate = random.uniform(prev_x - 3, prev_x + 3)
                # Accept with probability proportional to kernel density
                acceptance_prob = new_kernel_pdf(x_candidate) / new_kernel_pdf(prev_x)
                if random.random() < acceptance_prob:
                    return x_candidate
        
        # Set random seed for reproducibility
        random.seed(12)
        
        # Generate several random sample positions from the kernel
        new_random_sample_positions = []
        for _ in range(10):  # Generate 10 random samples from the kernel
            new_random_sample_positions.append(sample_from_new_kernel())
        new_random_sample_positions.append(next_target_x)  # End at the actual target value
        
        # Animate through the random positions sampled from the kernel along the temporary axis
        for i, x_pos in enumerate(new_random_sample_positions):
            # Position on the temporary axis (same y-coordinate as current chain point)
            temp_pos = ax.c2p(x_pos, 0)
            temp_pos[1] = current_chain_y
            
            if i < len(new_random_sample_positions) - 1:
                # Quick movements for intermediate positions
                self.play(new_sampling_dot.animate.move_to(temp_pos), run_time=0.15)
            else:
                # Slower final movement to target
                self.play(new_sampling_dot.animate.move_to(temp_pos), run_time=0.4)

        # Add label for the new proposed state x' (not x'')
        x_prime_label_new = Tex(r"x'", font_size=36)
        x_prime_label_new.set_color(TGREEN)
        x_prime_label_new.next_to(new_sampling_dot.get_center(), DOWN, buff=0.13)
        # x_prime_label_new.align_to(ax.c2p(0, -0.05), DOWN)
        self.play(Write(x_prime_label_new), run_time=0.3)

        self.wait(0.5)

        # Fade out the kernel for this iteration
        self.play(
            FadeOut(new_gaussian_kernel, run_time=0.5),
            FadeOut(new_gaussian_kernel_label, run_time=0.5),
        )

        self.wait(0.5)

        # endregion

        # region 18. Show the target density again for function evaluation -------------------
        # Show the target density again for function evaluation
        # This time, shift it to align with the temporary x-axis
        pdf_curve_new = ax.get_graph(
            lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) if NORMAL else
            0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x)),
            color=WHITE
        )
        # Shift the PDF curve to align with the temporary axis
        pdf_shift = current_chain_y - ax.c2p(0, 0)[1]  # Calculate shift from main axis
        pdf_curve_new.shift(UP * pdf_shift)
        self.plot_group.add(pdf_curve_new)
    
        self.play(ShowCreation(pdf_curve_new), run_time=2)

        # Define the target PDF function for evaluation (same as before)
        def target_pdf_new(x):
            if NORMAL:
                return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
            else:
                return 0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x))
        
        # Get the current positions and function values for the second iteration
        x_pos_new = prev_x  # Current state position (from sampling_dot)
        # Get the actual x-coordinate where the new_sampling_dot ended up
        x_prime_pos_new = ax.p2c(new_sampling_dot.get_center())[0]  # Use actual position of new_sampling_dot
        f_x_value_new = target_pdf_new(x_pos_new)
        f_x_prime_value_new = target_pdf_new(x_prime_pos_new)
        
        # Create vertical lines from the chain points to function values
        # For the current state (sampling_dot), draw from its chain position
        x_line_new = Line(
            sampling_dot.get_center(),  # Start from chain position
            np.array([ax.c2p(x_pos_new, f_x_value_new)[0], current_chain_y + f_x_value_new * ax.y_axis.get_unit_size(), 0]),  # End at shifted function value
            color=TBLUE,
            stroke_width=3
        )
        
        # For the proposed state, draw from the temporary axis level to function value
        x_prime_line_new = Line(
            new_sampling_dot.get_center(),  # Start from temporary axis position
            np.array([ax.c2p(x_prime_pos_new, f_x_prime_value_new)[0], current_chain_y + f_x_prime_value_new * ax.y_axis.get_unit_size(), 0]),  # End at shifted function value
            color=TGREEN,
            stroke_width=3
        )
        
        # Animate drawing both lines simultaneously
        self.play(
            ShowCreation(x_line_new),
            ShowCreation(x_prime_line_new),
            run_time=1.5
        )
        
        # Create function value labels (as before)
        f_x_label_new = Tex(f"f(x) = {f_x_value_new:.2f}", font_size=28)
        f_x_label_new.set_color(TBLUE)
        f_x_label_new.next_to(np.array([ax.c2p(x_pos_new, f_x_value_new)[0], current_chain_y + f_x_value_new * ax.y_axis.get_unit_size(), 0]), UP, buff=0.1)
        
        f_x_prime_label_new = Tex(f"f(x') = {f_x_prime_value_new:.2f}", font_size=28)
        f_x_prime_label_new.set_color(TGREEN)
        f_x_prime_label_new.next_to(np.array([ax.c2p(x_prime_pos_new, f_x_prime_value_new)[0], current_chain_y + f_x_prime_value_new * ax.y_axis.get_unit_size(), 0]), UP, buff=0.1)
        
        # Create dots at the function value points
        f_x_dot_new = Dot(np.array([ax.c2p(x_pos_new, f_x_value_new)[0], current_chain_y + f_x_value_new * ax.y_axis.get_unit_size(), 0]), radius=0.08, fill_color=TBLUE, fill_opacity=1.0)
        f_x_prime_dot_new = Dot(np.array([ax.c2p(x_prime_pos_new, f_x_prime_value_new)[0], current_chain_y + f_x_prime_value_new * ax.y_axis.get_unit_size(), 0]), radius=0.08, fill_color=TGREEN, fill_opacity=1.0)
        
        # Animate adding the labels and dots simultaneously (as before)
        self.play(
            Write(f_x_label_new),
            Write(f_x_prime_label_new),
            FadeIn(f_x_dot_new, scale=0.5),
            FadeIn(f_x_prime_dot_new, scale=0.5),
            pdf_curve_new.animate.set_stroke(opacity=0.3),  # Fade the function curve
            run_time=1.0
        )
        
        self.wait(1)

        # endregion

        # region 19. Update the rules box with the new function values and checkmark --------
        # Rule 2 applies (probabilistic acceptance) - show rule 2
        self.play(
            Write(rule2_condition),
            rule1_condition.animate.set_opacity(0.5),
            rule1_action.animate.set_opacity(0.5),
            run_time=1.0
        )
        self.play(
            Write(rule2_action),
            run_time=1.0
        )
        
        # After showing rule 2, move the function values to the rules box for calculation
        ratio_value = f_x_prime_value_new / f_x_value_new if f_x_value_new != 0 else float('inf')
        ratio_value = round(ratio_value, 2)  # Round to 2 decimal places
        
        # Position the calculation aligned with the rule2_action text
        calc_base_position = rule2_action.get_right() + RIGHT

        # Create a duplicate of the f(x')/f(x) fraction from rule2_action
        # Start it at the position of the fraction in rule2_action
        fraction_in_rule = rule2_action.get_part_by_tex(r"\frac{f(x')}{f(x)}")
        fraction_duplicate = Tex(r"\frac{f(x')}{f(x)}", font_size=18, color=WHITE)
        fraction_duplicate.move_to(fraction_in_rule.get_center())

        # Create the calculation components for vertical fraction (smaller size)
        fraction_line = Line(LEFT * 0.3, RIGHT * 0.3, stroke_width=1, color=WHITE)
        equals_1 = Tex("=", font_size=18, color=WHITE)
        equals_2 = Tex("=", font_size=18, color=WHITE)
        result = Tex(f"{ratio_value:.2f}", font_size=18, color=WHITE)
        
        # Calculate target positions for the calculation layout
        fraction_target_pos = calc_base_position
        
        # Position equals_1 to the right of the target fraction position
        equals_1.next_to(fraction_target_pos, RIGHT, buff=0.3)
        
        # Position the numeric fraction components after equals_1
        fraction_line.next_to(equals_1, RIGHT, buff=0.1)
        f_x_prime_target = fraction_line.get_center() + UP * 0.125  # Numerator above line
        f_x_target = fraction_line.get_center() + DOWN * 0.125      # Denominator below line
        
        # Position equals_2 and result to the right of the numeric fraction
        equals_2.next_to(fraction_line, RIGHT, buff=0.1)
        result.next_to(equals_2, RIGHT, buff=0.1)
        
        # # First show the duplicate f(x')/f(x) appearing and then moving to calculation area
        # self.play(
        #     fraction_duplicate.animate.move_to(fraction_target_pos),
        #     run_time=0.8
        # )
        
        # Animate the numbers from the labels moving to the calculation
        f_x_number_part = f_x_label_new.get_part_by_tex(f"{f_x_value_new:.2f}")
        f_x_prime_number_part = f_x_prime_label_new.get_part_by_tex(f"{f_x_prime_value_new:.2f}")
        
        self.play(
            fraction_duplicate.animate.move_to(fraction_target_pos),
            f_x_prime_number_part.animate.move_to(f_x_prime_target).scale(18/28),
            f_x_number_part.animate.move_to(f_x_target).scale(18/28),
            FadeOut(f_x_label_new.get_parts_by_tex("f(x) =")),
            FadeOut(f_x_prime_label_new.get_parts_by_tex("f(x') =")),
            FadeIn(equals_1),
            FadeIn(fraction_line),
            run_time=1.5
        )
        
        # Now add the rest of the calculation elements
        self.play(
            FadeIn(equals_2),
            FadeIn(result),
            run_time=1.0
        )
        
        self.wait(1)

        # endregion

        # region 20. Show random number generation and acceptance/rejection based on ratio
        # Create a number line from 0 to 1 below the rules box
        number_line_length = 3.0
        number_line_start = rules_box.get_bottom() + UP * 1.5 + RIGHT * 2.0
        number_line_end = number_line_start + RIGHT * number_line_length
        
        # Create the number line
        number_line = Line(number_line_start, number_line_end, color=WHITE, stroke_width=3)
        
        # Create tick marks and labels for 0 and 1
        tick_0 = Line(number_line_start + DOWN * 0.1, number_line_start + UP * 0.1, color=WHITE, stroke_width=2)
        tick_1 = Line(number_line_end + DOWN * 0.1, number_line_end + UP * 0.1, color=WHITE, stroke_width=2)
        
        label_0 = Tex("0", font_size=20, color=WHITE)
        label_0.next_to(tick_0, DOWN, buff=0.1)
        
        label_1 = Tex("1", font_size=20, color=WHITE)
        label_1.next_to(tick_1, DOWN, buff=0.1)
        
        # Show the number line
        self.play(
            ShowCreation(number_line),
            ShowCreation(tick_0),
            ShowCreation(tick_1),
            Write(label_0),
            Write(label_1),
            run_time=1.0
        )
        
        # Calculate position for the ratio_value (0.23) on the number line
        ratio_position = number_line_start + RIGHT * (ratio_value * number_line_length)
        
        # Create tick mark for ratio_value
        tick_ratio = Line(ratio_position + DOWN * 0.15, ratio_position + UP * 0.15, color=WHITE, stroke_width=2)
        
        # Duplicate the ratio value from the calculation and move it to the number line
        ratio_duplicate = result.copy()
        
        self.play(
            ShowCreation(tick_ratio),
            ratio_duplicate.animate.next_to(tick_ratio, DOWN, buff=0.1),
            run_time=1.0
        )
        
        # Create colored segments on the number line
        # Green segment (accept region): from 0 to ratio_value
        green_segment = Line(
            number_line_start, 
            ratio_position,
            color=TGREEN,
            stroke_width=8,
            stroke_opacity=0.8
        )
        
        # Red segment (reject region): from ratio_value to 1
        red_segment = Line(
            ratio_position,
            number_line_end,
            color=TRED,
            stroke_width=8,
            stroke_opacity=0.8
        )
        
        # Show the colored segments
        self.play(
            ShowCreation(green_segment),
            ShowCreation(red_segment),
            run_time=1.0
        )
        
        # Generate a random number that will be less than ratio_value for acceptance
        random.seed(42)
        final_random_value = random.uniform(0.05, ratio_value - 0.02)  # Ensure it's less than ratio_value
        
        # Create a dot that will move along the number line to show random sampling
        random_number_dot = Dot(radius=0.08, fill_color=YELLOW, fill_opacity=1.0)
        random_number_dot.move_to(number_line_start + UP * 0.3)  # Start above the number line at 0
        
        # Show the sampling dot
        self.play(FadeIn(random_number_dot, scale=0.5), run_time=0.5)
        
        # Animate the dot moving randomly along the number line several times before settling
        random_positions = []
        for _ in range(12):  # Generate several random positions
            random_positions.append(random.uniform(0.1, 0.9))
        random_positions.append(final_random_value)  # End at our target value
        
        # Animate through the random positions
        for i, pos in enumerate(random_positions):
            target_pos = number_line_start + RIGHT * (pos * number_line_length) + UP * 0.3
            if i < len(random_positions) - 1:
                # Quick movements for intermediate positions
                self.play(random_number_dot.animate.move_to(target_pos), run_time=0.1)
            else:
                # Slower final movement
                self.play(random_number_dot.animate.move_to(target_pos), run_time=0.3)
        

        
        # Since the random number is less than ratio_value, change the dot to green to show acceptance
        self.play(
            random_number_dot.animate.set_fill(TGREEN, opacity=1.0),
            run_time=0.5
        )
        
        # Add "ACCEPT!" text
        accept_result = Text("ACCEPT!", font="Gill Sans", font_size=24, color=TGREEN, weight=BOLD)
        accept_result.next_to(random_number_dot, UP, buff=0.2)
        
        self.play(Write(accept_result), run_time=0.8)
        
        self.wait(2)

        # endregion

        # region 21. Accept the second proposal and add to chain --------
        # Clean up the number line and function evaluation elements
        self.play(
            FadeOut(number_line),
            FadeOut(tick_0),
            FadeOut(tick_1),
            FadeOut(label_0),
            FadeOut(label_1),
            FadeOut(tick_ratio),
            FadeOut(ratio_duplicate),
            FadeOut(green_segment),
            FadeOut(red_segment),
            FadeOut(random_number_dot),
            FadeOut(x_line_new),
            FadeOut(x_prime_line_new),
            FadeOut(f_x_dot_new),
            FadeOut(f_x_prime_dot_new),
            FadeOut(x_prime_label_new),
            FadeOut(temp_x_axis),
            # Fade out the f(x')/f(x) calculation elements
            FadeOut(fraction_duplicate),
            FadeOut(equals_1),
            FadeOut(fraction_line),
            FadeOut(f_x_prime_number_part),
            FadeOut(f_x_number_part),
            FadeOut(equals_2),
            FadeOut(result),
            pdf_curve_new.animate.set_stroke(opacity=0.0),
            accept_result.animate.next_to(new_sampling_dot, UP, buff=0.2),
            rule1_condition.animate.set_opacity(1.0),
            rule1_action.animate.set_opacity(1.0),
            # Return the first point and line to normal opacity
            first_dot.animate.set_fill(opacity=0.8),
            chain_line.animate.set_stroke(opacity=0.8),
            run_time=1.5
        )
        
        self.wait(0.5)
        
        # Add the second accepted sample to the chain
        # The new point joins the existing chain structure from region 6
        # We just need to connect it to the existing chain and make it blue
        
        # Calculate where this point would be in the original extended chain from region 6
        # It would be the third-to-last point (index -3), so get its y-position
        total_samples = len(sample_xs)
        third_to_last_index = total_samples - 3  # Index of the third-to-last sample
        
        # Use the same chain extension logic from region 6 to find where this point should be
        if third_to_last_index == total_samples - 1:
            extension = 0.0  # Final point stays on axis
        else:
            extension_factor = (total_samples - third_to_last_index - 1) / (total_samples - 1)
            extension = max_extension * extension_factor
        
        # Get the axis position for the new sampling dot and apply the extension
        new_sampling_axis_pos = ax.c2p(ax.p2c(new_sampling_dot.get_center())[0], 0)
        new_sampling_target = new_sampling_axis_pos + chain_direction * extension
        
        # Create a connecting line from the current end of the chain to the new point
        # The current chain ends at sampling_dot, so connect from there to new_sampling_dot
        new_chain_line = Line(sampling_dot.get_center(), new_sampling_target)
        new_chain_line.set_stroke(TBLUE, width=3, opacity=0.8)
        
        # Move the new dot to its chain position and make it blue, then show the connection
        self.play(
            new_sampling_dot.animate.move_to(new_sampling_target).set_fill(TBLUE, opacity=0.8),
            ShowCreation(new_chain_line),
            x_label.animate.next_to(new_sampling_target, DOWN, buff=0.25),
            FadeOut(accept_result),
            run_time=1.5
        )
        
        self.wait(1)

        # endregion

        # region 22. Repeat the process for the next sample with rejection --------
        # Now we'll repeat the sampling process again, starting from the current state (new_sampling_dot)
        # This time the sample will be rejected to demonstrate the rejection case
        
        # Update the current position for the next iteration
        current_x_third = next_target_x  # The new_sampling_dot position becomes the new current state
        
        # Get the position of the current chain point for the temporary x-axis
        current_chain_pos_third = new_sampling_dot.get_center()
        current_chain_y_third = current_chain_pos_third[1]
        
        # Create a temporary x-axis aligned with the current chain point
        # Use screen coordinates directly
        left_point_third = ax.c2p(-4, 0)
        right_point_third = ax.c2p(4, 0)
        left_point_third[1] = current_chain_y_third  # Set y to match chain point
        right_point_third[1] = current_chain_y_third  # Set y to match chain point
        
        temp_x_axis_third = Line(
            left_point_third,
            right_point_third,
            color=WHITE,
            stroke_width=2,
            stroke_opacity=0.3
        )
        
        # Show the temporary x-axis
        self.play(ShowCreation(temp_x_axis_third), run_time=1.0)
        
        # Fade the previous iteration elements to lower opacity
        self.play(
            sampling_dot.animate.set_fill(opacity=0.3),
            new_chain_line.animate.set_stroke(opacity=0.3),
            first_dot.animate.set_fill(opacity=0.3),
            chain_line.animate.set_stroke(opacity=0.3),
            run_time=0.5
        )
        
        # Show the creation of a new Gaussian kernel centered at the new current state
        # But position it relative to the temporary axis
        third_gaussian_kernel = ax.get_graph(
            lambda x: np.exp(-(x-current_x_third)**2 / 2) / np.sqrt(2 * np.pi),
            color=TGREEN,
            x_range=[current_x_third-3, current_x_third+3]
        )
        # Shift the kernel to align with the temporary axis
        kernel_shift_third = current_chain_y_third - ax.c2p(0, 0)[1]  # Calculate shift from main axis
        third_gaussian_kernel.shift(UP * kernel_shift_third)
        
        third_gaussian_kernel_label = Text("Kernel", font="Gill Sans", font_size=36)
        third_gaussian_kernel_label.set_color(TGREEN)
        third_gaussian_kernel_label.next_to(third_gaussian_kernel, UP, buff=0.2)

        self.play(
            ShowCreation(third_gaussian_kernel),
            Write(third_gaussian_kernel_label),
            run_time=2
        )
        
        self.wait(1)

        # Create a new sampling dot for the third iteration on the temporary axis
        third_sampling_dot = Dot(radius=0.1, fill_color=TGREEN, fill_opacity=1.0)
        # Position it on the temporary x-axis at current_x_third - 3
        temp_axis_start_pos_third = ax.c2p(current_x_third - 3.0, 0)
        temp_axis_start_pos_third[1] = current_chain_y_third  # Align with temporary axis
        third_sampling_dot.move_to(temp_axis_start_pos_third)
        
        # Show the new sampling dot appearing
        self.play(FadeIn(third_sampling_dot, scale=0.5), run_time=0.5)
        
        # Define the third kernel function for sampling
        def third_kernel_pdf(x):
            return np.exp(-(x - current_x_third)**2 / 2) / np.sqrt(2 * np.pi)
        
        # Function to sample from the kernel using rejection sampling
        def sample_from_third_kernel():
            # Use rejection sampling for the Gaussian kernel
            while True:
                # Sample uniformly from the range
                x_candidate = random.uniform(current_x_third - 3, current_x_third + 3)
                # Accept with probability proportional to kernel density
                acceptance_prob = third_kernel_pdf(x_candidate) / third_kernel_pdf(current_x_third)
                if random.random() < acceptance_prob:
                    return x_candidate
        
        # Set random seed for reproducibility and choose a position in low probability area
        random.seed(16)
        
        # Generate several random sample positions from the kernel
        third_random_sample_positions = []
        for _ in range(10):  # Generate 10 random samples from the kernel
            third_random_sample_positions.append(sample_from_third_kernel())
        
        # For the final position, choose a value that will have low target density
        # This should be to the right of the current position where density is lower
        rejection_target_x = current_x_third + 1.5  # Move to the right where density is lower
        third_random_sample_positions.append(rejection_target_x)  # End at the rejection target
        
        # Animate through the random positions sampled from the kernel along the temporary axis
        for i, x_pos in enumerate(third_random_sample_positions):
            # Position on the temporary axis (same y-coordinate as current chain point)
            temp_pos = ax.c2p(x_pos, 0)
            temp_pos[1] = current_chain_y_third
            
            if i < len(third_random_sample_positions) - 1:
                # Quick movements for intermediate positions
                self.play(third_sampling_dot.animate.move_to(temp_pos), run_time=0.15)
            else:
                # Slower final movement to target
                self.play(third_sampling_dot.animate.move_to(temp_pos), run_time=0.4)

        # Add label for the new proposed state x' 
        x_prime_label_third = Tex(r"x'", font_size=36)
        x_prime_label_third.set_color(TGREEN)
        x_prime_label_third.next_to(third_sampling_dot.get_center(), DOWN, buff=0.13)
        self.play(Write(x_prime_label_third), run_time=0.3)

        self.wait(0.5)

        # Fade out the kernel for this iteration
        self.play(
            FadeOut(third_gaussian_kernel, run_time=0.5),
            FadeOut(third_gaussian_kernel_label, run_time=0.5),
        )

        self.wait(0.5)

        # endregion

        # region 23. Show the target density again for function evaluation (third iteration) -------------------
        # Show the target density again for function evaluation
        # This time, shift it to align with the third temporary x-axis
        pdf_curve_third = ax.get_graph(
            lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) if NORMAL else
            0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x)),
            color=WHITE
        )
        # Shift the PDF curve to align with the third temporary axis
        pdf_shift_third = current_chain_y_third - ax.c2p(0, 0)[1]  # Calculate shift from main axis
        pdf_curve_third.shift(UP * pdf_shift_third)
        self.plot_group.add(pdf_curve_third)
    
        self.play(ShowCreation(pdf_curve_third), run_time=2)

        # Define the target PDF function for evaluation (same as before)
        def target_pdf_third(x):
            if NORMAL:
                return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
            else:
                return 0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x))
        
        # Get the current positions and function values for the third iteration
        x_pos_third = current_x_third  # Current state position (from new_sampling_dot)
        # Get the actual x-coordinate where the third_sampling_dot ended up
        x_prime_pos_third = ax.p2c(third_sampling_dot.get_center())[0]  # Use actual position of third_sampling_dot
        f_x_value_third = target_pdf_third(x_pos_third)
        f_x_value_third = round(f_x_value_third, 3)  # Round to 3 decimal places
        f_x_prime_value_third = target_pdf_third(x_prime_pos_third)
        f_x_prime_value_third = round(f_x_prime_value_third, 3)  # Round to 3 decimal places
        
        # Create vertical lines from the chain points to function values
        # For the current state (new_sampling_dot), draw from its chain position
        x_line_third = Line(
            new_sampling_dot.get_center(),  # Start from chain position
            np.array([ax.c2p(x_pos_third, f_x_value_third)[0], current_chain_y_third + f_x_value_third * ax.y_axis.get_unit_size(), 0]),  # End at shifted function value
            color=TBLUE,
            stroke_width=3
        )
        
        # For the proposed state, draw from the temporary axis level to function value
        x_prime_line_third = Line(
            third_sampling_dot.get_center(),  # Start from temporary axis position
            np.array([ax.c2p(x_prime_pos_third, f_x_prime_value_third)[0], current_chain_y_third + f_x_prime_value_third * ax.y_axis.get_unit_size(), 0]),  # End at shifted function value
            color=TGREEN,
            stroke_width=3
        )
        
        # Animate drawing both lines simultaneously
        self.play(
            ShowCreation(x_line_third),
            ShowCreation(x_prime_line_third),
            run_time=1.5
        )
        
        # Create function value labels
        f_x_label_third = Tex(f"f(x) = {f_x_value_third:.3f}", font_size=28)
        f_x_label_third.set_color(TBLUE)
        f_x_label_third.next_to(np.array([ax.c2p(x_pos_third, f_x_value_third)[0], current_chain_y_third + f_x_value_third * ax.y_axis.get_unit_size(), 0]), UP, buff=0.1)
        
        f_x_prime_label_third = Tex(f"f(x') = {f_x_prime_value_third:.3f}", font_size=28)
        f_x_prime_label_third.set_color(TGREEN)
        f_x_prime_label_third.next_to(np.array([ax.c2p(x_prime_pos_third, f_x_prime_value_third)[0], current_chain_y_third + f_x_prime_value_third * ax.y_axis.get_unit_size(), 0]), UP, buff=0.1)
        
        # Create dots at the function value points
        f_x_dot_third = Dot(np.array([ax.c2p(x_pos_third, f_x_value_third)[0], current_chain_y_third + f_x_value_third * ax.y_axis.get_unit_size(), 0]), radius=0.08, fill_color=TBLUE, fill_opacity=1.0)
        f_x_prime_dot_third = Dot(np.array([ax.c2p(x_prime_pos_third, f_x_prime_value_third)[0], current_chain_y_third + f_x_prime_value_third * ax.y_axis.get_unit_size(), 0]), radius=0.08, fill_color=TGREEN, fill_opacity=1.0)
        
        # Animate adding the labels and dots simultaneously
        self.play(
            Write(f_x_label_third),
            Write(f_x_prime_label_third),
            FadeIn(f_x_dot_third, scale=0.5),
            FadeIn(f_x_prime_dot_third, scale=0.5),
            pdf_curve_third.animate.set_stroke(opacity=0.3),  # Fade the function curve
            run_time=1.0
        )
        
        self.wait(1)

        # endregion

        # region 24. Update the rules box with the third iteration function values and calculate ratio --------
        # Rule 2 applies (probabilistic acceptance) for this rejection scenario
        # Since rule 2 is already visible from the previous iteration, just update the opacity to highlight it
        self.play(
            rule2_condition.animate.set_opacity(1.0),
            rule2_action.animate.set_opacity(1.0),
            rule1_condition.animate.set_opacity(0.5),
            rule1_action.animate.set_opacity(0.5),
            run_time=1.0
        )
        
        # Calculate the ratio for the third iteration (this should be < 1 for rejection)
        ratio_value_third = f_x_prime_value_third / f_x_value_third if f_x_value_third != 0 else float('inf')
        ratio_value_third = round(ratio_value_third, 2)  # Round to 2 decimal places
        
        # Position the calculation aligned with the rule2_action text (same as before)
        calc_base_position_third = rule2_action.get_right() + RIGHT

        # Create a duplicate of the f(x')/f(x) fraction from rule2_action
        fraction_in_rule_third = rule2_action.get_part_by_tex(r"\frac{f(x')}{f(x)}")
        fraction_duplicate_third = Tex(r"\frac{f(x')}{f(x)}", font_size=18, color=WHITE)
        fraction_duplicate_third.move_to(fraction_in_rule_third.get_center())

        # Create the calculation components for vertical fraction (smaller size)
        fraction_line_third = Line(LEFT * 0.3, RIGHT * 0.3, stroke_width=1, color=WHITE)
        equals_1_third = Tex("=", font_size=18, color=WHITE)
        equals_2_third = Tex("=", font_size=18, color=WHITE)
        result_third = Tex(f"{ratio_value_third:.2f}", font_size=18, color=WHITE)
        
        # Calculate target positions for the calculation layout
        fraction_target_pos_third = calc_base_position_third
        
        # Position equals_1 to the right of the target fraction position
        equals_1_third.next_to(fraction_target_pos_third, RIGHT, buff=0.3)
        
        # Position the numeric fraction components after equals_1
        fraction_line_third.next_to(equals_1_third, RIGHT, buff=0.1)
        f_x_prime_target_third = fraction_line_third.get_center() + UP * 0.125  # Numerator above line
        f_x_target_third = fraction_line_third.get_center() + DOWN * 0.125      # Denominator below line
        
        # Position equals_2 and result to the right of the numeric fraction
        equals_2_third.next_to(fraction_line_third, RIGHT, buff=0.1)
        result_third.next_to(equals_2_third, RIGHT, buff=0.1)
        
        # Animate the numbers from the labels moving to the calculation
        f_x_number_part_third = f_x_label_third.get_part_by_tex(f"{f_x_value_third:.3f}")
        f_x_prime_number_part_third = f_x_prime_label_third.get_part_by_tex(f"{f_x_prime_value_third:.3f}")
        
        self.play(
            fraction_duplicate_third.animate.move_to(fraction_target_pos_third),
            f_x_prime_number_part_third.animate.move_to(f_x_prime_target_third).scale(18/28),
            f_x_number_part_third.animate.move_to(f_x_target_third).scale(18/28),
            FadeOut(f_x_label_third.get_parts_by_tex("f(x) =")),
            FadeOut(f_x_prime_label_third.get_parts_by_tex("f(x') =")),
            FadeIn(equals_1_third),
            FadeIn(fraction_line_third),
            run_time=1.5
        )
        
        # Now add the rest of the calculation elements
        self.play(
            FadeIn(equals_2_third),
            FadeIn(result_third),
            run_time=1.0
        )
        
        self.wait(1)

        # endregion

        # region 25. Show random number generation and rejection based on ratio --------
        # Create a number line from 0 to 1 below the rules box (same as region 20)
        number_line_length = 3.0
        number_line_start = rules_box.get_bottom() + UP * 1.5 + RIGHT * 2.0
        number_line_end = number_line_start + RIGHT * number_line_length
        
        # Create the number line
        number_line_third = Line(number_line_start, number_line_end, color=WHITE, stroke_width=3)
        
        # Create tick marks and labels for 0 and 1
        tick_0_third = Line(number_line_start + DOWN * 0.1, number_line_start + UP * 0.1, color=WHITE, stroke_width=2)
        tick_1_third = Line(number_line_end + DOWN * 0.1, number_line_end + UP * 0.1, color=WHITE, stroke_width=2)
        
        label_0_third = Tex("0", font_size=20, color=WHITE)
        label_0_third.next_to(tick_0_third, DOWN, buff=0.1)
        
        label_1_third = Tex("1", font_size=20, color=WHITE)
        label_1_third.next_to(tick_1_third, DOWN, buff=0.1)
        
        # Show the number line
        self.play(
            ShowCreation(number_line_third),
            ShowCreation(tick_0_third),
            ShowCreation(tick_1_third),
            Write(label_0_third),
            Write(label_1_third),
            run_time=1.0
        )
        
        # Calculate position for the ratio_value_third on the number line
        ratio_position_third = number_line_start + RIGHT * (ratio_value_third * number_line_length)
        
        # Create tick mark for ratio_value_third
        tick_ratio_third = Line(ratio_position_third + DOWN * 0.15, ratio_position_third + UP * 0.15, color=WHITE, stroke_width=2)
        
        # Duplicate the ratio value from the calculation and move it to the number line
        ratio_duplicate_third = result_third.copy()
        
        self.play(
            ShowCreation(tick_ratio_third),
            ratio_duplicate_third.animate.next_to(tick_ratio_third, DOWN, buff=0.1),
            run_time=1.0
        )
        
        # Create colored segments on the number line
        # Green segment (accept region): from 0 to ratio_value_third
        green_segment_third = Line(
            number_line_start, 
            ratio_position_third,
            color=TGREEN,
            stroke_width=8,
            stroke_opacity=0.8
        )
        
        # Red segment (reject region): from ratio_value_third to 1
        red_segment_third = Line(
            ratio_position_third,
            number_line_end,
            color=TRED,
            stroke_width=8,
            stroke_opacity=0.8
        )
        
        # Show the colored segments
        self.play(
            ShowCreation(green_segment_third),
            ShowCreation(red_segment_third),
            run_time=1.0
        )
        
        # Generate a random number that will be GREATER than ratio_value_third for rejection
        random.seed(24)
        final_random_value_third = random.uniform(ratio_value_third + 0.05, 0.95)  # Ensure it's greater than ratio_value_third
        
        # Create a dot that will move along the number line to show random sampling
        random_number_dot_third = Dot(radius=0.08, fill_color=YELLOW, fill_opacity=1.0)
        random_number_dot_third.move_to(number_line_start + UP * 0.3)  # Start above the number line at 0
        
        # Show the sampling dot
        self.play(FadeIn(random_number_dot_third, scale=0.5), run_time=0.5)
        
        # Animate the dot moving randomly along the number line several times before settling
        random_positions_third = []
        for _ in range(12):  # Generate several random positions
            random_positions_third.append(random.uniform(0.1, 0.9))
        random_positions_third.append(final_random_value_third)  # End at our target value for rejection
        
        # Animate through the random positions
        for i, pos in enumerate(random_positions_third):
            target_pos = number_line_start + RIGHT * (pos * number_line_length) + UP * 0.3
            if i < len(random_positions_third) - 1:
                # Quick movements for intermediate positions
                self.play(random_number_dot_third.animate.move_to(target_pos), run_time=0.1)
            else:
                # Slower final movement
                self.play(random_number_dot_third.animate.move_to(target_pos), run_time=0.3)
        
        # Since the random number is GREATER than ratio_value_third, change the dot to red to show rejection
        self.play(
            random_number_dot_third.animate.set_fill(TRED, opacity=1.0),
            run_time=0.5
        )
        
        # Add "REJECT!" text
        reject_result = Text("REJECT!", font="Gill Sans", font_size=24, color=TRED, weight=BOLD)
        reject_result.next_to(random_number_dot_third, UP, buff=0.2)
        
        self.play(Write(reject_result), run_time=0.8)
        
        self.wait(2)

        # endregion

        # region 26. Handle rejection - clean up and show chain staying at current position --------
        # Clean up the number line and function evaluation elements
        self.play(
            FadeOut(number_line_third),
            FadeOut(tick_0_third),
            FadeOut(tick_1_third),
            FadeOut(label_0_third),
            FadeOut(label_1_third),
            FadeOut(tick_ratio_third),
            FadeOut(ratio_duplicate_third),
            FadeOut(green_segment_third),
            FadeOut(red_segment_third),
            FadeOut(random_number_dot_third),
            FadeOut(x_line_third),
            FadeOut(x_prime_line_third),
            FadeOut(f_x_dot_third),
            FadeOut(f_x_prime_dot_third),
            FadeOut(temp_x_axis_third),
            # Fade out the f(x')/f(x) calculation elements
            FadeOut(fraction_duplicate_third),
            FadeOut(equals_1_third),
            FadeOut(fraction_line_third),
            FadeOut(f_x_prime_number_part_third),
            FadeOut(f_x_number_part_third),
            FadeOut(equals_2_third),
            FadeOut(result_third),
            pdf_curve_third.animate.set_stroke(opacity=0.0),
            reject_result.animate.next_to(third_sampling_dot, UP, buff=0.2),
            rule1_condition.animate.set_opacity(1.0),
            rule1_action.animate.set_opacity(1.0),
            run_time=1.5
        )
        
        self.wait(0.5)
        
        # Make the rejected sample "poof" away with a scaling animation
        self.play(
            third_sampling_dot.animate.scale(0.1).set_fill(opacity=0.0),
            x_prime_label_third.animate.scale(0.1).set_fill(opacity=0.0),
            reject_result.animate.scale(0.1).set_fill(opacity=0.0),
            run_time=0.8
        )
        
        # Remove the rejected elements completely
        self.remove(third_sampling_dot, x_prime_label_third, reject_result)
        
        # Since we rejected, the chain stays at the current position (new_sampling_dot)
        # We need to create a new sample dot at the same x-position as new_sampling_dot
        # Calculate where this point would be in the original extended chain from region 6
        # It would be the fourth-to-last point (one more than the previous addition)
        total_samples = len(sample_xs)
        fourth_to_last_index = total_samples - 4  # Index of the fourth-to-last sample
        
        # Use the same chain extension logic from region 6 to find where this point should be
        if fourth_to_last_index == total_samples - 1:
            extension = 0.0  # Final point stays on axis
        else:
            extension_factor = (total_samples - fourth_to_last_index - 1) / (total_samples - 1)
            extension = max_extension * extension_factor
        
        # Get the axis position for the new sampling dot (same x as current state) and apply the extension
        current_x_pos = ax.p2c(new_sampling_dot.get_center())[0]  # Get x-coordinate of current state
        rejected_sample_axis_pos = ax.c2p(current_x_pos, 0)  # Same x-position as current state
        rejected_sample_target = rejected_sample_axis_pos + chain_direction * extension
        
        # Create a new dot that represents the "rejected iteration" (stays at same x)
        rejected_iteration_dot = Dot(radius=0.1, fill_color=TBLUE, fill_opacity=0.8)
        # Start it at the current chain position (new_sampling_dot) and move it to the new chain position
        rejected_iteration_dot.move_to(new_sampling_dot.get_center())
        
        # Create a connecting line from the current end of the chain to the new position
        # The current chain ends at new_sampling_dot, so connect from there to the new position
        rejected_chain_line = Line(new_sampling_dot.get_center(), rejected_sample_target)
        rejected_chain_line.set_stroke(TBLUE, width=3, opacity=0.8)
        
        # Show the new dot appearing at the current position
        self.play(FadeIn(rejected_iteration_dot, scale=0.5), run_time=0.5)
        
        # Move the dot to its new chain position and show the connection
        # Also return the previous elements to normal opacity
        self.play(
            rejected_iteration_dot.animate.move_to(rejected_sample_target),
            ShowCreation(rejected_chain_line),
            x_label.animate.next_to(rejected_sample_target, DOWN, buff=0.25),
            # Return previous elements to normal opacity
            sampling_dot.animate.set_fill(opacity=0.8),
            new_chain_line.animate.set_stroke(opacity=0.8),
            first_dot.animate.set_fill(opacity=0.8),
            chain_line.animate.set_stroke(opacity=0.8),
            run_time=1.5
        )
        
        self.wait(1)

        # endregion

        # region 27. Rapidly sample the next 40 points in the chain --------
        # We'll continue backwards through the chain, showing 40 more samples
        # For each sample, we'll briefly highlight which rule applies
        
        self.play(FadeOut(x_label), run_time=0.5)  # Fade out the x label

        # Get the remaining sample positions from the original chain
        # We've already shown samples at indices -1, -2, -3, -4, so start from -5
        remaining_samples = sample_xs[-45:-4]  # Get 41 more samples (we'll use 40 of them)
        remaining_samples = remaining_samples[::-1]  # Reverse to go backwards through the chain
        
        # Track the current state for the rapid sampling
        current_rapid_x = current_x_pos  # Start from the current x position
        current_rapid_dot = rejected_iteration_dot  # Start from the current chain end
        current_rapid_line = rejected_chain_line  # Start from the current chain line
        
        # Define the target PDF function for quick evaluation
        def rapid_target_pdf(x):
            if NORMAL:
                return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
            else:
                return 0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x))
        
        # Iterate through the next 40 samples
        for i, next_x in enumerate(remaining_samples[:40]):
            # Calculate function values
            f_current = rapid_target_pdf(current_rapid_x)
            f_next = rapid_target_pdf(next_x)
            
            # Determine which rule applies
            if f_next > f_current:
                # Rule 1 applies - certain acceptance
                rule_to_highlight = rule1_condition
                rule_action_to_highlight = rule1_action
                rule_to_fade = rule2_condition
                rule_action_to_fade = rule2_action
            else:
                # Rule 2 applies - probabilistic acceptance
                rule_to_highlight = rule2_condition
                rule_action_to_highlight = rule2_action
                rule_to_fade = rule1_condition
                rule_action_to_fade = rule1_action
            
            # Calculate where this new dot should be positioned in the chain
            # Use the correct index from the original sample_xs array
            # We're currently at index -4, and going backwards through the chain
            current_sample_index = -5 - i  # -5, -6, -7, etc.
            absolute_index = len(sample_xs) + current_sample_index  # Convert to positive index
            
            # Calculate extension for the new sample using the original chain logic
            # Make sure we don't go below 0 for the absolute_index
            if absolute_index <= 0 or absolute_index >= len(sample_xs) - 1:
                extension = 0.0  # Points at the ends stay on axis
            else:
                extension_factor = (len(sample_xs) - absolute_index - 1) / (len(sample_xs) - 1)
                extension = max_extension * extension_factor
            
            # Get the axis position for the new sample and apply the extension
            new_sample_axis_pos = ax.c2p(next_x, 0)
            new_sample_target = new_sample_axis_pos + chain_direction * extension
            
            # Create the new dot
            new_rapid_dot = Dot(radius=0.1, fill_color=TBLUE, fill_opacity=0.8)
            new_rapid_dot.move_to(new_sample_target)
            
            # Create the connecting line
            new_rapid_line = Line(current_rapid_dot.get_center(), new_sample_target)
            new_rapid_line.set_stroke(TBLUE, width=3, opacity=0.8)
            
            
            # Animate: highlight rule, add dot and line, then fade rule back
            animations_list = [
                rule_to_highlight.animate.set_opacity(1.0),
                rule_action_to_highlight.animate.set_opacity(1.0),
                rule_to_fade.animate.set_opacity(0.3),
                rule_action_to_fade.animate.set_opacity(0.3),
                FadeIn(new_rapid_dot, scale=0.5),
                ShowCreation(new_rapid_line),
            ]
            
            # Run the animations quickly
            self.play(*animations_list, run_time=0.1)
            
            # # Quickly fade the rule highlighting back to normal and remove label
            # fade_animations = [
            #     rule_to_highlight.animate.set_opacity(0.5),
            #     rule_action_to_highlight.animate.set_opacity(0.5),
            #     rule_to_fade.animate.set_opacity(0.5),
            #     rule_action_to_fade.animate.set_opacity(0.5),
            # ]
            
            # self.play(*fade_animations, run_time=0.15)
            
            # Update current state for next iteration
            current_rapid_x = next_x
            current_rapid_dot = new_rapid_dot
            current_rapid_line = new_rapid_line
        
        # After all rapid sampling, return rules to normal state
        self.play(
            rule1_condition.animate.set_opacity(1.0),
            rule1_action.animate.set_opacity(1.0),
            rule2_condition.animate.set_opacity(1.0),
            rule2_action.animate.set_opacity(1.0),
            run_time=0.5
        )
        
        self.wait(1)

        # endregion

        # region 28. Add the rest of the chain and collapse back to x-axis --------
        # First, add the rest of the chain (the remaining samples that weren't animated)
        # We've animated up to sample index -45 (from the rapid sampling), so add the rest
        remaining_chain_samples = sample_xs[:-45]  # All samples before the animated ones
        
        # Create dots and lines for the remaining chain using the same logic as region 6
        remaining_dots = []
        remaining_lines = []
        
        # Start from the beginning of the chain - create the first dot without a connection
        prev_dot = None
        
        for i, x_val in enumerate(remaining_chain_samples):
            # Calculate the extension for this sample
            sample_index = i  # Index in the original sample_xs array
            if sample_index == len(sample_xs) - 1:
                extension = 0.0  # Final point stays on axis
            else:
                extension_factor = (len(sample_xs) - sample_index - 1) / (len(sample_xs) - 1)
                extension = max_extension * extension_factor
            
            # Create the dot at its extended chain position
            sample_axis_pos = ax.c2p(x_val, 0)
            sample_target = sample_axis_pos + chain_direction * extension
            
            new_dot = Dot(radius=0.1, fill_color=TBLUE, fill_opacity=0.8)
            new_dot.move_to(sample_target)
            remaining_dots.append(new_dot)
            
            # Create the connecting line from the previous dot (if there is one)
            if prev_dot is not None:
                new_line = Line(prev_dot.get_center(), sample_target)
                new_line.set_stroke(TBLUE, width=3, opacity=0.8)
                remaining_lines.append(new_line)
            
            prev_dot = new_dot
        
        # Now connect the last remaining dot to the first dot of the animated chain
        # We need to connect the last dot of remaining_dots (sample -45) to sample -44
        # Sample -44 is represented by the last dot from rapid sampling (current_rapid_dot)
        if remaining_dots:
            connection_line = Line(remaining_dots[-1].get_center(), current_rapid_dot.get_center())
            connection_line.set_stroke(TBLUE, width=3, opacity=0.8)
            remaining_lines.append(connection_line)
        
        # Add all the remaining chain elements to the scene instantly (no animation needed since off-screen)
        if remaining_dots:
            self.add(*remaining_dots)
        
        if remaining_lines:
            self.add(*remaining_lines)
        
        # Now collect all chain elements for collapse
        chain_elements_to_collapse = []
        chain_lines_to_remove = []
        
        # Add the main chain dots we know about
        try:
            chain_elements_to_collapse.append(first_dot)
        except: pass
        try:
            chain_elements_to_collapse.append(sampling_dot)
        except: pass
        try:
            chain_elements_to_collapse.append(new_sampling_dot)
        except: pass
        try:
            chain_elements_to_collapse.append(rejected_iteration_dot)
        except: pass
        try:
            chain_elements_to_collapse.append(current_rapid_dot)  # The last dot from rapid sampling
        except: pass
        
        # Add all the remaining dots we just created
        chain_elements_to_collapse.extend(remaining_dots)
        
        # Add the main chain lines
        try:
            chain_lines_to_remove.append(chain_line)
        except: pass
        try:
            chain_lines_to_remove.append(new_chain_line)
        except: pass
        try:
            chain_lines_to_remove.append(rejected_chain_line)
        except: pass
        try:
            chain_lines_to_remove.append(current_rapid_line)  # The last line from rapid sampling
        except: pass
        
        # Add all the remaining lines we just created
        chain_lines_to_remove.extend(remaining_lines)
        
        # More aggressive search for all blue dots and lines in the scene
        # This should capture all the rapid sampling elements that weren't explicitly tracked
        x_axis_y = ax.c2p(0, 0)[1]
        
        for obj in self.mobjects:
            if isinstance(obj, Dot) and obj not in chain_elements_to_collapse:
                try:
                    center = obj.get_center()
                    # Check if it's positioned above the x-axis (part of the chain)
                    if center[1] > x_axis_y + 0.05:  # Above the x-axis with small buffer
                        fill_color = obj.get_fill_color()
                        # More robust blue color check
                        try:
                            # Convert color to numpy array if it isn't already
                            import numpy as np
                            color_array = np.array(fill_color)
                            if len(color_array) >= 3:
                                # Check if blue component (index 2) is dominant
                                if color_array[2] > max(color_array[0], color_array[1]) * 0.7:
                                    chain_elements_to_collapse.append(obj)
                        except:
                            # Fallback: if it's above the axis, it's probably a chain element
                            chain_elements_to_collapse.append(obj)
                except Exception as e:
                    pass
            elif isinstance(obj, Line) and obj not in chain_lines_to_remove:
                try:
                    # Check if it's a line positioned above the x-axis
                    line_center = (obj.get_start() + obj.get_end()) / 2
                    if line_center[1] > x_axis_y + 0.05:  # Above the x-axis
                        stroke_color = obj.get_stroke_color()
                        # More robust blue color check
                        try:
                            import numpy as np
                            color_array = np.array(stroke_color)
                            if len(color_array) >= 3:
                                # Check if blue component (index 2) is dominant
                                if color_array[2] > max(color_array[0], color_array[1]) * 0.7:
                                    chain_lines_to_remove.append(obj)
                        except:
                            # Fallback: if it's above the axis, it's probably a chain element
                            chain_lines_to_remove.append(obj)
                except Exception as e:
                    pass
        
        # Create animations to move all dots to the x-axis and collapse lines
        dot_animations = []
        for i, dot in enumerate(chain_elements_to_collapse):
            try:
                # Get the x-coordinate of the current dot position
                current_pos = dot.get_center()
                # Move to x-axis at the same x-coordinate
                target_pos = ax.c2p(ax.p2c(current_pos)[0], 0)
                dot_animations.append(dot.animate.move_to(target_pos))
            except Exception as e:
                pass
        
        # Instead of fading lines, collapse them to the x-axis
        line_collapse_animations = []
        for i, line in enumerate(chain_lines_to_remove):
            try:
                # Get the start and end points of the line
                start_pos = line.get_start()
                end_pos = line.get_end()
                
                # Calculate target positions on the x-axis
                start_x = ax.p2c(start_pos)[0]
                end_x = ax.p2c(end_pos)[0]
                target_start = ax.c2p(start_x, 0)
                target_end = ax.c2p(end_x, 0)
                
                # Create a new line that will be the target for this line's animation
                target_line = Line(target_start, target_end)
                target_line.set_stroke(TBLUE, width=3, opacity=0.8)
                
                # Animate the line transforming to the collapsed position
                line_collapse_animations.append(Transform(line, target_line))
            except Exception as e:
                pass
        
        # Execute the collapse animation (without fading out the rules box)
        all_animations = dot_animations + line_collapse_animations
        
        if all_animations:
            self.play(
                *all_animations,
                run_time=2.0,
                rate_func=smooth
            )
        else:
            # Fallback if no animations found
            self.wait(2.0)
        
        # After the collapse, fade out the collapsed lines on the x-axis
        collapsed_lines_to_remove = []
        for line in chain_lines_to_remove:
            try:
                collapsed_lines_to_remove.append(line)
            except:
                pass
        
        if collapsed_lines_to_remove:
            self.play(
                *[FadeOut(line) for line in collapsed_lines_to_remove],
                run_time=0.5
            )
        
        # Update the sample_dots list to include all collapsed dots
        self.sample_dots = chain_elements_to_collapse
        
        self.wait(1)

        # endregion

        # region 29. Grow histogram bars from x-axis and redraw target density --------
        # Now that all chain points are collapsed to the x-axis, grow histogram bars upward
        # and redraw the target density curve
        
        # First, collect all the x-coordinates of the collapsed dots
        sample_x_positions = []
        for dot in self.sample_dots:
            try:
                x_coord = ax.p2c(dot.get_center())[0]
                sample_x_positions.append(x_coord)
            except:
                pass
        
        # Create histogram data - use the same approach as regions 3-5
        hist_range = (-4, 4)
        bin_edges = np.linspace(hist_range[0], hist_range[1], 21)  # 20 equal-width bins
        hist_counts, _ = np.histogram(sample_x_positions, bins=bin_edges)
        
        # Normalize the histogram to match the density scale (area = 1)
        total_samples = len(sample_x_positions)
        
        # Create histogram bars using the same logic as region 5
        histogram_bars = VGroup()
        for j, c in enumerate(hist_counts):
            if c == 0:
                continue
            x_left = bin_edges[j]
            x_right = bin_edges[j + 1]
            # Normalize the bar heights so that total area is 1
            bar_width = x_right - x_left
            bar_height = c / (bar_width * total_samples)
            
            # Create rectangle for histogram bar
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
            histogram_bars.add(bar)
        
        # Animate the histogram bars growing from the x-axis
        # Start with bars at zero height
        for bar in histogram_bars:
            bar.stretch_to_fit_height(0.01)  # Very small initial height
            bar.align_to(ax.c2p(0, 0), DOWN)  # Align bottom to x-axis
        
        # Add bars to scene
        self.add(histogram_bars)
        
        # Animate bars growing to full height
        # Recreate the target bars with full height
        target_bars = VGroup()
        for j, c in enumerate(hist_counts):
            if c == 0:
                continue
            x_left = bin_edges[j]
            x_right = bin_edges[j + 1]
            bar_width = x_right - x_left
            bar_height = c / (bar_width * total_samples)
            
            target_bar = Rectangle(
                width=bar_width * ax.x_axis.get_unit_size(),
                height=bar_height * ax.y_axis.get_unit_size(),
                fill_color=WHITE,
                fill_opacity=0.3,
                stroke_width=1,
                stroke_color=WHITE
            )
            target_bar.move_to(ax.c2p((x_left + x_right) / 2, bar_height / 2))
            target_bars.add(target_bar)
        
        self.wait(1)
        
        # Now redraw the target density curve
        target_density = ax.get_graph(
            lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) if NORMAL else
            0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x)),
            color=WHITE,
            stroke_width=3
        )
        # Don't add to plot_group since we want to fade it out separately in region 30
        
        # Animate the target density appearing
        self.play(
            Transform(histogram_bars, target_bars),
            ShowCreation(target_density),
            run_time=2.0,
            rate_func=smooth
        )
        
        self.wait(2)

        # endregion

        # region 30. Fade out plot elements and scale up rules box to center --------
        # Collect all plot elements to fade out
        plot_elements_to_fade = []
        
        # Add the main plot group elements
        if hasattr(self, 'plot_group'):
            plot_elements_to_fade.extend(self.plot_group.submobjects)
        
        # Add the axes
        plot_elements_to_fade.append(ax)
        
        # Add any histogram bars that are still visible
        try:
            plot_elements_to_fade.append(histogram_bars)
        except:
            pass
        
        # Add any remaining sample dots on the x-axis
        if hasattr(self, 'sample_dots'):
            plot_elements_to_fade.extend(self.sample_dots)
        
        # Add any other plot-related elements that might be visible
        # Search for any remaining visible elements that are part of the plot
        for obj in self.mobjects:
            try:
                # Check if it's a plot-related object (positioned in the main plot area)
                center = obj.get_center()
                # If it's in the main plot area (roughly y between -1 and 3, x between -6 and 6)
                if (-6 <= center[0] <= 6 and -1 <= center[1] <= 3 and 
                    obj not in plot_elements_to_fade and 
                    obj != rules_box and obj != rules_title and 
                    obj != rule1_condition and obj != rule1_action and 
                    obj != rule2_condition and obj != rule2_action):
                    # Don't fade the rules box and its contents
                    plot_elements_to_fade.append(obj)
            except:
                pass
        
        # Fade out all plot elements
        fade_animations = [FadeOut(element) for element in plot_elements_to_fade 
                          if element != rules_box and element != rules_title and 
                          element != rule1_condition and element != rule1_action and 
                          element != rule2_condition and element != rule2_action]
        
        # Scale up the text elements and move to center
        text_scale_factor = 2.0  # Scale text to double size
        screen_center = ORIGIN  # Center of the screen
        
        # Scale the rules box differently - make it wider
        box_width_scale = 3.0   # Make it wider
        box_height_scale = 1.8  # Keep height scaling more modest
        
        # Create individual text animations to center each element as it scales
        text_animations = [
            rules_title.animate.scale(text_scale_factor).move_to(screen_center + UP * 1.8),
            rule1_condition.animate.scale(text_scale_factor).move_to(screen_center + UP * 1.0),
            rule1_action.animate.scale(text_scale_factor).move_to(screen_center + UP * 0.3),
            rule2_condition.animate.scale(text_scale_factor).move_to(screen_center + DOWN * 0.8),
            rule2_action.animate.scale(text_scale_factor).move_to(screen_center + DOWN * 1.5)
        ]
        
        box_animations = [
            rules_box.animate.stretch_to_fit_width(rules_box.get_width() * box_width_scale)
                             .stretch_to_fit_height(rules_box.get_height() * box_height_scale)
                             .move_to(screen_center)
        ]
        
        # Execute all animations simultaneously
        all_animations = fade_animations + text_animations + box_animations
        
        self.play(
            *all_animations,
            run_time=1
        )
        
        self.wait(1)

        # endregion

        # region 31. Highlight the first rule --------
        self.play(
            rule2_condition.animate.set_opacity(0.3),
            rule2_action.animate.set_opacity(0.3),
        )

        rule1_transform_condition = Tex(r"\text{If the proposed sample has \textbf{higher} density:}", font_size=36)
        rule1_transform_condition.move_to(rule1_condition.get_center())
        self.play(Transform(rule1_condition, rule1_transform_condition))

        x_prime_part = rule1_action.get_part_by_tex("x'")
        x_prime_part_transform = Tex(r"\text{it!}", font_size=40)
        x_prime_part_transform.move_to(x_prime_part.get_center() + 0.02 * DOWN)
        self.play(Transform(x_prime_part, x_prime_part_transform))

        self.play(
            rule1_condition.animate.set_opacity(0.3),
            rule1_action.animate.set_opacity(0.3),
            # x_prime_part_transform.animate.set_opacity(0.3),
            x_prime_part.animate.set_opacity(0.3),
            rule2_condition.animate.set_opacity(1.0),
            rule2_action.animate.set_opacity(1.0),
            run_time=0.5
        )

        rule2_transform_condition = Tex(r"\text{If the proposed sample has \textbf{lower} density:}", font_size=36)
        rule2_transform_condition.move_to(rule2_condition.get_center())
        self.play(Transform(rule2_condition, rule2_transform_condition))

        # frac_part = rule2_action.get_part_by_tex(r"\frac{f(x')}{f(x)}")
        # non_frac_part = rule2_action.get_part_by_tex(r"\text{Accept with prob. }")
        # frac_part_transform = Tex(r"\text{ proportional to how much lower}", font_size=36)
        # frac_part_transform.next_to(non_frac_part, RIGHT, buff=0.1)
        # new_condition = VGroup(non_frac_part, frac_part_transform)
        # # new_condition.move_to(rule2_action.get_center())
        # self.play(Transform(frac_part, frac_part_transform),
        #         #   non_frac_part.animate.move_to(non_frac_part.get_center()),
        #           new_condition.animate.move_to(rule2_action.get_center())
        # )
        rule2_transform_action = Tex(r"\text{Accept with prob proportional to how much lower}", font_size=36)
        rule2_transform_action.move_to(rule2_action.get_center())
        self.play(Transform(rule2_action, rule2_transform_action))

        self.play(
            rule1_condition.animate.set_opacity(1.0),
            rule1_action.animate.set_opacity(1.0),
            x_prime_part.animate.set_opacity(1.0),
            run_time=0.5
        )

        # endregion

        # region 32. More intuition --------
        rule1 = VGroup(rule1_condition, rule1_action, x_prime_part)
        text1 = Text("Sample in high density regions.", font="Gill Sans", font_size=46)
        text1.set_color(TGREEN)
        text1.move_to(rule1.get_center())
        self.play(rule1.animate.set_opacity(0.0),
                  Write(text1),
                  run_time=0.5)

        rule2 = VGroup(rule2_transform_condition, rule2_transform_action)
        text2 = Text("Sometimes still sample in low density regions.", font="Gill Sans", font_size=46)
        text2.set_color(TYELLOW)
        text2.move_to(rule2.get_center())
        self.play(rule2_transform_condition.animate.set_opacity(0.0),
                  rule2_transform_action.animate.set_opacity(0.0),
                  rule2_condition.animate.set_opacity(0.0),
                  rule2_action.animate.set_opacity(0.0),
                  Write(text2),
                  run_time=0.5)

        # endregion

        # region 33. Fade everything out and clear scene --------
        # Collect all visible objects to fade out
        all_objects_to_fade = []
        
        # Add the rules box and all text elements
        try:
            all_objects_to_fade.append(rules_box)
        except:
            pass
        
        try:
            all_objects_to_fade.append(text1)
        except:
            pass
            
        try:
            all_objects_to_fade.append(text2)
        except:
            pass
        
        # Add any remaining visible objects in the scene
        for obj in self.mobjects:
            if obj not in all_objects_to_fade:
                all_objects_to_fade.append(obj)
        
        # Fade out everything
        if all_objects_to_fade:
            self.play(
                *[FadeOut(obj) for obj in all_objects_to_fade],
                run_time=2.0
            )
        
        # Clear the scene completely
        self.clear()
        
        self.wait(1)
        
        # endregion

        # region 34. Convergence in the limit --------
        theorem = Text("MCMC will converge to the target distribution in the limit of infinite samples!", font="Gill Sans", font_size=36)
        self.play(Write(theorem))
        self.play(theorem.animate.move_to(3*UP))

        # Create the axis and target density using the same styling as region 1
        ax_conv = Axes(
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
        ax_conv.move_to(DOWN * 0.4)

        # Create a group to hold plot elements (same as region 1)
        plot_group_conv = VGroup()
        plot_group_conv.add(ax_conv)
        self.add(plot_group_conv)

        # Draw the target density (same as region 2)
        pdf_curve_conv = ax_conv.get_graph(
            lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) if NORMAL else
            0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x)),
            color=WHITE
        )
        plot_group_conv.add(pdf_curve_conv)
        self.play(ShowCreation(pdf_curve_conv), run_time=2)

        # Create containers for scatter & histogram (same as region 3)
        scatter_dots_conv = VGroup()
        histogram_bars_conv = VGroup()
        plot_group_conv.add(scatter_dots_conv, histogram_bars_conv)

        # Define sample sizes to demonstrate convergence
        sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]  # Much fewer sample sizes

        # Generate MCMC samples for demonstration using the same function as earlier
        np.random.seed(42)  # For reproducibility
        
        # Define the target PDF function (same as used earlier in the file)
        def target_pdf_conv(x):
            if NORMAL:
                return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
            else:
                return 0.4 * np.exp(-x**2 / 2) * (1 + 0.5 * np.sin(3 * x))
        
        # Use the mcmc_sampling function to generate proper MCMC samples
        all_samples = mcmc_sampling(target_pdf_conv, nsamples=sample_sizes[-1], xinit=0.0)

        # Set up histogram bins (same as region 4)
        bin_edges_conv = np.linspace(-4, 4, 21)  # 20 equal-width bins

        # Create a sample count label with separate "Samples:" and number components
        samples_text = Text("Samples: ", font="Gill Sans", font_size=36, color=TBLUE)
        samples_text.set_color(TBLUE)
        samples_number = Text("0", font="Gill Sans", font_size=36, color=TBLUE)
        samples_number.next_to(samples_text, RIGHT, buff=0.1)
        samples_number.set_color(TBLUE)
        sample_count_label = VGroup(samples_text, samples_number)
        
        # Position in upper left of the plot area
        sample_count_label.move_to(ax_conv.c2p(-2, 0.45))
        self.play(Write(sample_count_label), run_time=0.5)

        # Animate through different sample sizes (similar to region 5)
        for i, n_samples in enumerate(sample_sizes):
            # Get the first n_samples from our generated data
            current_samples = all_samples[:n_samples]
            
            # Update only the number part of the sample count label
            new_samples_number = Text(f"{n_samples:,}", font="Gill Sans", font_size=36, color=TBLUE)
            new_samples_number.set_color(TBLUE)
            new_samples_number.next_to(samples_text, RIGHT, buff=0.1)
            
            # Calculate histogram once using numpy (much faster)
            hist_counts, _ = np.histogram(current_samples, bins=bin_edges_conv)
            total_samples = len(current_samples)
            
            # Create new scatter dots - only show a subset for performance
            new_scatter_dots = VGroup()
            # Show every 10th dot to reduce visual clutter and improve performance
            sample_step = max(1, len(current_samples) // 500)  # Show at most 500 dots
            for j, x in enumerate(current_samples[::sample_step]):
                dot = Dot(ax_conv.c2p(x, 0), radius=0.08, fill_color=TBLUE, fill_opacity=0.6)
                new_scatter_dots.add(dot)

            # Rebuild histogram bars (same logic as region 5)
            new_bars_conv = VGroup()
            for j, c in enumerate(hist_counts):
                if c == 0:
                    continue
                x_left = bin_edges_conv[j]
                x_right = bin_edges_conv[j+1]
                # Normalize the bar heights so that total area is 1
                bar_width = x_right - x_left
                bar_height = c / (bar_width * total_samples)

                # Create rectangle for histogram bar
                bar = Rectangle(
                    width=bar_width * ax_conv.x_axis.get_unit_size(),
                    height=bar_height * ax_conv.y_axis.get_unit_size(),
                    fill_color=WHITE,
                    fill_opacity=0.3,
                    stroke_width=1,
                    stroke_color=WHITE
                )
                # Position the bar correctly
                bar.move_to(ax_conv.c2p((x_left + x_right) / 2, bar_height / 2))
                new_bars_conv.add(bar)

            # Animate the transition to the new histogram and dots
            if i == 0:
                # First iteration - show everything
                self.play(
                    Transform(samples_number, new_samples_number),
                    FadeIn(new_scatter_dots),
                    Transform(histogram_bars_conv, new_bars_conv),
                    run_time=0.5
                )
                scatter_dots_conv = new_scatter_dots
            else:
                # Subsequent iterations - transform to new state
                self.play(
                    Transform(samples_number, new_samples_number),
                    Transform(scatter_dots_conv, new_scatter_dots),
                    Transform(histogram_bars_conv, new_bars_conv),
                    run_time=0.5
                )
            
            # Brief pause to observe the histogram
            self.wait(0.25)
        
        # endregion

        # region 35. Start of practical considerations --------
        # Fade out the convergence plot elements and move theorem back to just above center
        prob_text = Text("Pretty Cool! Just one problem...", font="Gill Sans", font_size=36, color=TBLUE)
        prob_text.set_color(TBLUE)
        prob_text.move_to(0.25 * DOWN)

        self.play(
            FadeOut(plot_group_conv),
            FadeOut(sample_count_label),
            FadeOut(scatter_dots_conv),
            theorem.animate.move_to(0.25 * UP),
            run_time=1.5
        )

        self.play(
            Write(prob_text),
            run_time=1.0
        )

        # Highlight "in the limit of infinite samples" in the theorem
        theorem_parts = theorem.get_parts_by_text("in the limit of infinite samples!")
        if theorem_parts:
            # Create a box around the highlighted text
            # Get the bounding box of all the theorem parts
            if len(theorem_parts) > 0:
                # Calculate the bounding rectangle for all parts
                left = min([part.get_left()[0] for part in theorem_parts])
                right = max([part.get_right()[0] for part in theorem_parts])
                bottom = min([part.get_bottom()[1] for part in theorem_parts])
                top = max([part.get_top()[1] for part in theorem_parts])
                
                # Add some padding
                padding = 0.1
                highlight_box = Rectangle(
                    width=right - left + 2 * padding,
                    height=top - bottom + 2 * padding,
                    stroke_color=YELLOW,
                    stroke_width=2,
                    fill_opacity=0.0
                )
                highlight_box.move_to([(left + right) / 2, (top + bottom) / 2, 0])
                
                self.play(
                    *[theorem_part.animate.set_color(YELLOW) for theorem_part in theorem_parts],
                    ShowCreation(highlight_box),
                    lag_ratio=0.25,
                    run_time=2.0
                )

        # Move current text to the top to make room for the clock action
        # Calculate the current vertical separation between the texts
        current_separation = theorem.get_center()[1] - prob_text.get_center()[1]

        # Make the highlight box disappear instantly
        self.remove(highlight_box)
        
        self.play(
            theorem.animate.move_to(2.5 * UP).set_opacity(0.3),
            prob_text.animate.move_to(2.5 * UP - current_separation * UP).set_opacity(0.3),
            run_time=1.0
        )

        # Create a clock for the silly animation
        clock_radius = 1.2  # Increased from 0.8
        clock_circle = Circle(radius=clock_radius, color=WHITE, stroke_width=4)  # Thicker stroke
        clock_circle.move_to(ORIGIN)  # Center of screen
        
        # Clock hands (hour and minute) - bigger
        hour_hand = Line(ORIGIN, UP * 0.6, color=WHITE, stroke_width=6)  # Increased from 0.4 and stroke 4
        minute_hand = Line(ORIGIN, UP * 0.9, color=WHITE, stroke_width=4)  # Increased from 0.6 and stroke 2
        hour_hand.move_to(clock_circle.get_center())
        minute_hand.move_to(clock_circle.get_center())
        
        # Clock numbers (12, 3, 6, 9) - bigger
        twelve = Text("12", font_size=28, color=WHITE)  # Increased from 20
        twelve.move_to(clock_circle.get_center() + UP * 0.8)  # Increased from 0.55
        three = Text("3", font_size=28, color=WHITE)
        three.move_to(clock_circle.get_center() + RIGHT * 0.8)
        six = Text("6", font_size=28, color=WHITE)
        six.move_to(clock_circle.get_center() + DOWN * 0.8)
        nine = Text("9", font_size=28, color=WHITE)
        nine.move_to(clock_circle.get_center() + LEFT * 0.8)
        
        clock_numbers = VGroup(twelve, three, six, nine)
        clock_group = VGroup(clock_circle, hour_hand, minute_hand, clock_numbers)
        
        # Create the "nobody has time" text 
        infinite_text = Text("Nobody has time for infinite samples!", font="Gill Sans", font_size=44, color=ORANGE)
        infinite_text.next_to(clock_group, DOWN, buff=0.5)  # Position it below the clock

        # Show the clock
        self.play(
            FadeIn(clock_group),
            Write(infinite_text),
            run_time=1.0  # Changed back from 0.1
        )
        
        # Make the clock tick fast (minute hand spinning rapidly)
        self.play(
            Rotate(minute_hand, angle=8*TAU, about_point=clock_circle.get_center()),
            Rotate(hour_hand, angle=2*TAU/3, about_point=clock_circle.get_center()),
            run_time=3.0,
            rate_func=linear
        )
        
        self.wait(1.0)
        
        # Fade out all text elements on screen
        self.play(
            FadeOut(theorem),
            FadeOut(prob_text),
            FadeOut(clock_group),
            FadeOut(infinite_text),
            run_time=0.5
        )

        # endregion

        # region 36. Practical considerations --------
        finite_text = Text("We can still make a finite sample size work well!", font="Gill Sans", font_size=36)
        self.play(
            Write(finite_text),
            run_time=1.0
        )

        trick_text = Text("Sometimes we just need a few extra tricks!", font="Gill Sans", font_size=36)
        trick_text.set_color(TYELLOW)
        trick_text.move_to(0.25 * DOWN)
        self.play(
            Write(trick_text),
            finite_text.animate.move_to(0.25 * UP),
            run_time=1.0
        )

        # endregion

        # Idea: to illustrate these tricks, let's sample from a trickier target distribution
        # pick a target density with a smaller region of high probability

        