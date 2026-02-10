# utilities.py
import io
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from PIL import Image, ImageOps


# ---------- Core logic (no widgets / notebook globals) ----------

def update(grid, birth_rule={3}, survive_rule={2, 3}, mask=None):
    """
    Computes the next state of the grid according to the given rules.
    Optionally applies a mask: only cells where mask is True will update.
    """
    n = (
        np.roll(grid, 1, 0) + np.roll(grid, -1, 0) +
        np.roll(grid, 1, 1) + np.roll(grid, -1, 1) +
        np.roll(np.roll(grid, 1, 0), 1, 1) +
        np.roll(np.roll(grid, 1, 0), -1, 1) +
        np.roll(np.roll(grid, -1, 0), 1, 1) +
        np.roll(np.roll(grid, -1, 0), -1, 1)
    )

    # Apply rules
    birth = np.isin(n, list(birth_rule)) & (grid == 0)
    survive = np.isin(n, list(survive_rule)) & (grid == 1)

    next_state = np.zeros_like(grid)
    next_state[birth | survive] = 1

    if mask is not None:
        # If mask is True, use next_state; if False, keep grid.
        return np.where(mask, next_state, grid)
    else:
        return next_state


def get_quantized_layers(img, num_colors=4):
    """
    Quantizes the image to `num_colors` and returns:
    - layers: list of binary grids (one per color)
    - colors: list of (R,G,B) tuples normalized to 0-1
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Quantize to a palette image
    quantized = img.quantize(colors=num_colors, method=Image.MAXCOVERAGE)

    # Extract palette
    p = quantized.getpalette()
    palette_colors = []
    for i in range(num_colors):
        r = p[i * 3]
        g = p[i * 3 + 1]
        b = p[i * 3 + 2]
        palette_colors.append((r / 255.0, g / 255.0, b / 255.0))

    arr = np.array(quantized)
    layers = []
    for i in range(num_colors):
        layers.append((arr == i).astype(int))

    return layers, palette_colors


def run_simulation_reverse(grid, steps=30, rules='conway', update_mode='uniform'):
    """
    Runs the automaton forward for `steps` and returns the reverse sequence.
    """
    history = [grid.copy()]
    current_grid = grid

    # Define rule sets (Birth, Survive)
    rule_sets = {
        'conway': ({3}, {2, 3}),
        'anneal': ({4, 6, 7, 8}, {3, 5, 6, 7, 8}),
        'highlife': ({3, 6}, {2, 3}),
        'daynight': ({3, 6, 7, 8}, {3, 4, 6, 7, 8}),
        'maze': ({3}, {1, 2, 3, 4, 5}),
        'seeds': ({2}, {}),  # Explosive
        '2x2': ({3, 6}, {1, 2, 5}),
        'replicator': ({1, 3, 5, 7}, {1, 3, 5, 7}),
    }

    b_rule, s_rule = rule_sets.get(rules, rule_sets['conway'])
    rows, cols = grid.shape

    for i in range(steps):
        # Determine mask based on update_mode
        mask = None
        if update_mode == 'patch':
            mask = np.zeros((rows, cols), dtype=bool)
            h = np.random.randint(rows // 5, rows // 2)
            w = np.random.randint(cols // 5, cols // 2)
            y = np.random.randint(0, rows - h)
            x = np.random.randint(0, cols - w)
            mask[y:y + h, x:x + w] = True
        elif update_mode == 'noise' or update_mode == 'noise_50':
            mask = np.random.random((rows, cols)) < 0.5
        elif update_mode == 'noise_10':
            mask = np.random.random((rows, cols)) < 0.1
        elif update_mode == 'noise_90':
            mask = np.random.random((rows, cols)) < 0.9
        elif update_mode == 'noise_ramp':
            # Ramp probability from 0.05 to 0.95 over the steps
            prob = 0.05 + (0.9 * (i / max(1, steps - 1)))
            mask = np.random.random((rows, cols)) < prob

        current_grid = update(current_grid, b_rule, s_rule, mask=mask)
        history.append(current_grid.copy())

    # Return ONLY the reverse sequence
    return history[::-1]


def create_gif(all_layer_frames, palette, filename="gol_output.gif", duration=100):
    """
    Combines layers into frames and saves as a GIF.
    """
    if not all_layer_frames:
        return None

    num_layers = len(all_layer_frames)
    num_frames = len(all_layer_frames[0])
    h, w = all_layer_frames[0][0].shape
    
    images = []
    
    for f in range(num_frames):
        combined = np.zeros((h, w, 3))
        for i in range(num_layers):
            mask = all_layer_frames[i][f]
            color = palette[i]
            colored_layer = np.zeros((h, w, 3))
            colored_layer[mask == 1] = color
            combined += colored_layer
            
        combined = np.clip(combined, 0, 1)
        img_uint8 = (combined * 255).astype(np.uint8)
        images.append(Image.fromarray(img_uint8))
        
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    return filename


# ---------- UI builder (widgets + callback) ----------

class GOLImageApp:
    """
    An application to run a multi-layer Game of Life simulation from an image.
    """
    def __init__(self, grid_size=128, steps=60):
        self.grid_size = grid_size
        self.steps = steps
        self.ui = self.build_ui()

    def build_ui(self):
        """
        Builds and displays the widget UI for the reverse simulation.
        Returns the top-level widget container.
        """
        self.uploader = widgets.FileUpload(
            accept='image/*',
            multiple=False,
            description='Upload Image'
        )

        self.rule_dropdown = widgets.Dropdown(
            options=[
                ('Anneal (Gradual Morphing)', 'anneal'),
                ('Conway (Standard)', 'conway'),
                ('Maze (Structure)', 'maze'),
                ('HighLife (Replicators)', 'highlife'),
                ('Day & Night', 'daynight'),
                ('Seeds (Explosive)', 'seeds'),
                ('2x2', '2x2')
            ],
            value='anneal',
            description='Rule:'
        )

        self.update_mode_dropdown = widgets.Dropdown(
            options=[
                ('Uniform (All cells update)', 'uniform'),
                ('Random Patches (Glitchy)', 'patch'),
                ('Noise: Low (10%)', 'noise_10'),
                ('Noise: Medium (50%)', 'noise_50'),
                ('Noise: High (90%)', 'noise_90'),
                ('Noise: Ramp (Low -> High)', 'noise_ramp')
            ],
            value='uniform',
            description='Update:'
        )

        self.colors_slider = widgets.IntSlider(
            value=6,
            min=2,
            max=16,
            step=1,
            description='# Colors:'
        )

        self.steps_slider = widgets.IntSlider(
            value=self.steps,
            min=10,
            max=300,
            step=10,
            description='Steps:'
        )

        run_btn = widgets.Button(description="Run & Reverse")
        run_btn.on_click(self.on_run_clicked)
        # Use a VBox as a placeholder for results instead of an Output widget
        self.results_container = widgets.VBox([])

        ui = widgets.VBox([
            widgets.Label("1. Upload an image"),
            self.uploader,
            widgets.HBox([self.rule_dropdown, self.update_mode_dropdown]),
            widgets.HBox([self.colors_slider, self.steps_slider]),
            widgets.Label("2. Run Simulation & Show Reverse"),
            run_btn,
            self.results_container
        ])
        return ui

    def on_run_clicked(self, b):
        # Clear the previous results and show a loading message
        # Use an Output widget to capture prints/errors during processing
        log_output = widgets.Output()
        self.results_container.children = [widgets.Label("Processing..."), log_output]
        
        with log_output:
            try:
                if not self.uploader.value:
                    print("Please upload an image first!")
                    return

                # Extract file content from FileUpload widget
                try:
                    # Handle ipywidgets 8.x (tuple of dicts) vs 7.x (dict of dicts)
                    uploaded_value = self.uploader.value
                    if isinstance(uploaded_value, tuple):
                        # ipywidgets 8.x
                        file_info = uploaded_value[0]
                    else:
                        # ipywidgets 7.x
                        # value is a dict {filename: {content: ...}}
                        file_info = next(iter(uploaded_value.values()))
                    
                    content = file_info['content']
                    # Ensure content is bytes (it might be memoryview)
                    if hasattr(content, 'tobytes'):
                        content = content.tobytes()
                except Exception as e:
                    print(f"Could not read upload: {e}")
                    return

                # Load and resize image
                try:
                    img = Image.open(io.BytesIO(content))
                    
                    # Fix orientation based on EXIF tags
                    img = ImageOps.exif_transpose(img)
                    
                    resample = getattr(Image, 'Resampling', Image).LANCZOS
                    img = img.resize((self.grid_size, self.grid_size), resample)
                except Exception as e:
                    print(f"Error loading image: {e}")
                    return

                # Quantize and split into layers
                num_colors = self.colors_slider.value
                steps_val = self.steps_slider.value
                print(f"Quantizing to {num_colors} colors...")
                layers, palette = get_quantized_layers(img, num_colors)

                print(
                    f"Simulating {num_colors} layers for {steps_val} steps with rule "
                    f"'{self.rule_dropdown.value}' and mode '{self.update_mode_dropdown.value}'..."
                )

                # Simulate each layer
                all_layer_frames = []
                for layer_grid in layers:
                    frames = run_simulation_reverse(
                        layer_grid,
                        steps=steps_val,
                        rules=self.rule_dropdown.value,
                        update_mode=self.update_mode_dropdown.value
                    )
                    all_layer_frames.append(frames)

                # Store for GIF export
                self.current_frames = all_layer_frames
                self.current_palette = palette

                total_frames = len(all_layer_frames[0])
                
                # Pre-calculate combined frames to prevent playback lag/jitter
                print("Pre-rendering frames for smooth playback...")
                self.rendered_frames = []
                h, w = layers[0].shape
                for f in range(total_frames):
                    combined = np.zeros((h, w, 3))
                    for i in range(num_colors):
                        mask = all_layer_frames[i][f]
                        color = palette[i]
                        colored_layer = np.zeros((h, w, 3))
                        colored_layer[mask == 1] = color
                        combined += colored_layer
                    self.rendered_frames.append(np.clip(combined, 0, 1))

                # --- Display Logic ---
                play = widgets.Play(
                    value=0,
                    min=0,
                    max=total_frames - 1,
                    step=1,
                    interval=100,
                    description="Press play",
                )

                slider = widgets.IntSlider(
                    value=0,
                    min=0,
                    max=total_frames - 1,
                    step=1,
                    description="Frame"
                )

                widgets.jslink((play, 'value'), (slider, 'value'))

                # Speed Control
                speed_slider = widgets.IntSlider(
                    value=200,
                    min=50,
                    max=1000,
                    step=50,
                    description='Delay (ms):',
                    layout=widgets.Layout(width='250px')
                )
                widgets.jslink((speed_slider, 'value'), (play, 'interval'))

                # Create figure
                # Use plt.ioff() to prevent auto-display in notebook
                with plt.ioff():
                    fig, ax = plt.subplots(figsize=(6, 6))
                    if hasattr(fig.canvas, 'header_visible'):
                        fig.canvas.header_visible = False
                    if hasattr(fig.canvas, 'footer_visible'):
                        fig.canvas.footer_visible = False
                    ax.axis('off')

                # Display first frame with fixed scaling to prevent flickering
                im = ax.imshow(self.rendered_frames[0], vmin=0, vmax=1)
                ax.set_title(f"Reverse Playback: Step {steps_val}")
                
                def update_view(change):
                    frame_idx = change['new']
                    # Use pre-rendered frame
                    im.set_data(self.rendered_frames[frame_idx])
                    ax.set_title(f"Reverse Playback: Step {steps_val - frame_idx}")
                    fig.canvas.draw_idle()

                slider.observe(update_view, names='value')

                # GIF Button & Controls
                save_gif_btn = widgets.Button(
                    description="Save GIF",
                    icon="file-image-o",
                    tooltip="Save animation as .gif"
                )
                save_gif_btn.on_click(self.on_save_gif_clicked)

                # Slider for GIF frame duration
                gif_duration_slider = widgets.IntSlider(
                    value=100,
                    min=20,
                    max=500,
                    step=10,
                    description='GIF Delay (ms):',
                    style={'description_width': 'initial'},
                    layout=widgets.Layout(width='200px')
                )
                self.gif_duration_slider = gif_duration_slider

                # Create the results widget
                # Using fig.canvas directly for ipympl support
                results_widget = widgets.VBox([
                    fig.canvas,
                    widgets.HBox([play, slider]),
                    widgets.HBox([speed_slider]),
                    widgets.HBox([gif_duration_slider, save_gif_btn])
                ])

                # Set the results widget as the children of our container
                self.results_container.children = [results_widget]
            except Exception as e:
                print(f"An error occurred: {e}")
                # Keep log_output visible

    def on_save_gif_clicked(self, b):
        if not hasattr(self, 'current_frames'):
            return
            
        b.disabled = True
        original_desc = b.description
        b.description = "Saving..."
        
        try:
            filename = "gol_reverse.gif"
            # Get duration from slider, default 100 if not found
            duration_val = self.gif_duration_slider.value if hasattr(self, 'gif_duration_slider') else 100
            
            create_gif(self.current_frames, self.current_palette, filename=filename, duration=duration_val)
            b.description = f"Saved: {filename}"
        except Exception as e:
            b.description = "Error"
            print(f"GIF creation failed: {e}")
        finally:
            b.disabled = False
