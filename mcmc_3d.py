from manimlib import *
import numpy as np
import random
import matplotlib.pyplot as plt
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.three_dimensions import SurfaceMesh

TBLUE = "#80B9FF"  # Deep Sky Blue
TPURPLE = "#C3B8FF"
TGREEN = "#80CFB9"
TPINK = "#FFA4E7"
TRED = "#FAB0AE"
TYELLOW = "#FEEAAD"


class MCMC3d(Scene):
    def construct(self):
        # region 1: Set up 3D axes
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[0, 0.25, 0.1],
            width=8,
            height=8,
            depth=4,
        )
        axes.set_width(10)
        axes.center()

        self.frame.reorient(43, 64, 1, IN, 10)
        self.add(axes)
        # endregion

        # region 2: Define the PDF surface
        # def pdf_func(u, v):
        #     # This function should return just the z-value (height)
        #     return 0.16 * np.exp(-0.5 * (u**2 + v**2)) * (1 + 0.5 * np.sin(3 * u) * np.cos(3 * v))

        # gaussian mixture
        def pdf_func(u, v):
            return 0.3 * np.exp(-1.0 * ((u - 1)**2 + (v - 1)**2)) + 0.2 * np.exp(-1.0 * ((u + 1)**2 + (v + 1)**2))

        # Use axes.get_graph to automatically align with coordinate system
        surface = axes.get_graph(
            pdf_func,
            u_range=[-3.5, 3.5],
            v_range=[-3.5, 3.5],
            resolution=(150, 150),
            opacity=0.8,
            color=WHITE
        )

        self.add(surface)
        
        # Add a mesh to show the surface structure
        surface_mesh = SurfaceMesh(surface, resolution=(25, 25))
        surface_mesh.set_stroke(WHITE, 0.5, opacity=0.1)
        self.add(surface_mesh)

        # Create contour lines on the xy-plane
        contour_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
        contour_lines = VGroup()
        
        for level in contour_levels:
            # Create points for contour at this level
            u_vals = np.linspace(-3.5, 3.5, 100)
            v_vals = np.linspace(-3.5, 3.5, 100)
            U, V = np.meshgrid(u_vals, v_vals)
            Z = np.array([[pdf_func(u, v) for u in u_vals] for v in v_vals])
            
            # Find contour lines using matplotlib's contour algorithm
            cs = plt.contour(U, V, Z, levels=[level])
            plt.close()  # Close the matplotlib figure
            
            # Extract contour paths and create Manim curves
            level_curves = VGroup()
            
            # Get the contour paths from the QuadContourSet
            for path_collection in cs.allsegs[0]:  # allsegs[0] contains paths for the first (and only) level
                if len(path_collection) > 2:  # Only process paths with enough points
                    # Convert to 3D points on xy-plane (z=0)
                    points_3d = [axes.c2p(x, y, 0) for x, y in path_collection]
                    if len(points_3d) > 1:
                        # Create a smooth curve through the points
                        curve = VMobject()
                        curve.set_points_smoothly(points_3d)
                        curve.set_stroke(TBLUE, width=2, opacity=0)
                        level_curves.add(curve)
            
            contour_lines.add(level_curves)
        
        # Add contour lines to scene (initially invisible)
        self.add(contour_lines)

        self.play(FadeIn(surface, run_time=0.5),
                  FadeIn(surface_mesh, run_time=0.5),
                  self.frame.animate.reorient(30, 70, 1, IN, 10), run_time=3)
        
        # Animate surface becoming transparent while contour lines fade in
        self.play(
            surface.animate.set_opacity(0.1),
            contour_lines.animate.set_stroke(opacity=0.8),
            run_time=2
        )
        # self.play(FadeOut(surface_mesh, run_time=0.5),
        #           FadeOut(surface, run_time=0.5))
        # self.play(self.frame.animate.reorient(0, 180, 0, IN, 10), run_time=3)
        # endregion

        # region 3: MCMC chain animation
        def mcmc_sampling(target_pdf, nsamples=500, xinit=np.array([0.0, 0.0]), scale_factor=1.0):
            x = xinit
            samples = [x]
            
            for i in range(nsamples):
                x_new = x + [2 * np.random.normal(0, scale_factor), 2 * np.random.normal(0, scale_factor)]
                pdf_current = target_pdf(*x)
                pdf_new = target_pdf(*x_new)
                if pdf_current > 0:
                    acceptance_prob = min(1.0, pdf_new / pdf_current)
                else:
                    acceptance_prob = 1.0 if pdf_new > 0 else 0.0
                if np.random.random() < acceptance_prob:
                    x = x_new
                samples.append(x)
            return samples
        
        np.random.seed(4)
        samples = mcmc_sampling(pdf_func, nsamples=500, xinit=np.array([0.0, 0.0]), scale_factor=0.5)
        sample_dots = VGroup()
        chain_lines = VGroup()
        
        # Keep track of recent chain connections for fading effect
        recent_lines = []
        max_visible_lines = 3  # Number of recent connections to keep visible
        
        # Store rotation parameters for smooth independent rotation
        rotation_start_time = self.time
        rotation_duration = 90  # 30 seconds for one full rotation
        initial_theta = 30
        
        # Animate each sample appearing one by one (much faster approach)
        for i, sample in enumerate(samples[:300]):  # Fewer samples for speed
            x, y = sample
            z = 0  # Dots appear on the xy-plane
            
            # Create and add the sample dot
            dot = Dot(point=axes.c2p(x, y, z), radius=0.1, fill_opacity=0.8)
            dot.set_color(TBLUE)
            sample_dots.add(dot)
            
            # Create connection line to previous sample
            if i > 0:
                prev_sample = samples[i-1]
                prev_x, prev_y = prev_sample
                
                # Create line connecting previous sample to current sample
                line = Line(
                    start=axes.c2p(prev_x, prev_y, z),
                    end=axes.c2p(x, y, z),
                    stroke_width=2,
                    color=TBLUE
                )
                line.set_opacity(0.8)
                chain_lines.add(line)
                recent_lines.append(line)
                
                # Remove oldest lines if we have too many
                if len(recent_lines) > max_visible_lines:
                    old_line = recent_lines.pop(0)
                    chain_lines.remove(old_line)
                    self.remove(old_line)

            # Update camera rotation smoothly and independently
            current_time = self.time
            elapsed_time = current_time - rotation_start_time
            rotation_progress = (elapsed_time % rotation_duration) / rotation_duration
            current_theta = initial_theta + 360 * rotation_progress
            self.frame.reorient(current_theta, 70, 1, IN, 10)
            
            if i < 1000: # i % 10 == 0 or i < 100:
                # Determine animation speed based on sample number
                if i < 10:  # First 10 samples: slow and clear
                    run_time = 0.001
                    highlight_time = 0.001
                elif i < 50:  # Next 40 samples: medium speed
                    run_time = 0.001
                    highlight_time = 0.001
                else:  # Remaining samples: very fast
                    run_time = 0.001
                    highlight_time = 0.001

                # Prepare animations
                animations = [FadeIn(dot)]
                
                # Add chain line animation if this isn't the first sample
                if i > 0 and len(recent_lines) > 0:
                    current_line = recent_lines[-1]
                    animations.append(FadeIn(current_line))
                    
                    # Fade out older lines gradually
                    for j, old_line in enumerate(recent_lines[:-1]):
                        opacity = 0.8 * (j + 1) / len(recent_lines)
                        animations.append(old_line.animate.set_opacity(opacity))

                # animate: drop dot and chain line, then highlight
                highlight_dot = dot.copy().set_fill(TBLUE, opacity=0.8).scale(1.5)
                animations.append(FadeIn(highlight_dot, scale=0.5))
                
                self.play(*animations, run_time=run_time)

                self.play(
                    FadeOut(highlight_dot, scale=1.5),
                    run_time=highlight_time
                    )
            else:
                self.add(dot)
                if i > 0:
                    self.add(recent_lines[-1])
                for j, old_line in enumerate(recent_lines[:-1]):
                        opacity = 0.8 * (j + 1) / len(recent_lines)
                        old_line.set_opacity(opacity)

        # Pause to observe the final result
        self.wait(1)
        # endregion

        # region 4
        self.play(FadeOut(surface),
                  FadeOut(surface_mesh),
                  self.frame.animate.reorient(90, 0, 0, IN, 10), run_time=2)
        
        # # Show how the histogram approximates the surface
        # self.play(
        #     surface.animate.set_opacity(0.9),
        #     surface_mesh.animate.set_stroke(WHITE, 0.3, opacity=0.8),
        #     run_time=2
        # )
        # self.wait(2)
        
        #endregion


