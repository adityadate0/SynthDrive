import gradio as gr
import os
import time
import threading
import queue
from PIL import Image
from src.processor import ImageProcessor, AutoCaptioner
from src.generator import SyntheticGenerator
from create_showcase import create_triptych

# --- SETUP ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "models")
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# Initialize Engine (CPU Mode for Best Quality)
print(r"""
   _____             __  __    ___       _            
  / ___/11  ______  / /_/ /_  / _ \_____(_)  _____    
  \__ \/ / / / __ \/ __/ __ \/ / / / ___/ / | / / _ \   
 ___/ / /_/ / / / / /_/ / / / /_/ / /  / /| |/ /  __/   
/____/\__, /_/_/_/\__/_/ /_/_____/_/  /_/ |___/\___/    
     /____/  DATA FACTORY v1.0                          
""")
print("--- 🚀 Initializing SynthDrive Engine... ---")
generator = SyntheticGenerator(use_cpu=True) 
captioner = AutoCaptioner(model_dir, use_cpu=True)
print("--- ✅ Engine Ready! Launching GUI... ---")

# --- CUSTOM CSS & JS ---
# 1. Hide footer to prevent theme toggling/resetting
# 2. Force Dark Mode colors via JS
custom_css = """
footer {visibility: hidden !important;} 
.gradio-container { max-width: 950px !important; margin: auto; }
h1, h2 { text-align: center; }
"""

# JavaScript to force Dark Mode immediately on load
js_func = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

def process_stream(input_image, instruction):
    if input_image is None:
        return None, None, "❌ Please upload an image first."
    
    # 1. Setup
    input_pil = Image.fromarray(input_image).convert("RGB")
    temp_input_path = os.path.join(output_dir, "temp_input.png")
    input_pil.save(temp_input_path)

    # 2. Vision
    yield None, None, "👁️ AI is seeing..."
    scene_desc = captioner.generate_caption(temp_input_path)
    full_prompt = f"{scene_desc}, {instruction}, clean car body, glossy finish, photorealistic, 8k, cinematic lighting"
    
    # 3. Geometry
    yield None, None, "📐 Locking Edges..."
    _, edges = ImageProcessor.get_canny_edges(temp_input_path)
    temp_edge_path = os.path.join(output_dir, "temp_edges.png")
    edges.save(temp_edge_path)

    # 4. LIVE GENERATION
    yield None, None, "🧬 Starting Generator (This will take time)..."
    
    # Queue to pass images from thread to GUI
    img_queue = queue.Queue()
    
    def on_step(image_preview):
        img_queue.put(image_preview)

    def run_gen():
        result = generator.generate(full_prompt, edges, step_callback=on_step)
        img_queue.put("DONE")
        img_queue.put(result)

    # Start the heavy work in background thread
    gen_thread = threading.Thread(target=run_gen)
    gen_thread.start()

    # Watch the queue and update GUI live
    final_result = None
    while gen_thread.is_alive():
        try:
            preview = img_queue.get(timeout=0.5)
            if preview == "DONE":
                break
            # Update LIVE PREVIEW box
            yield preview, None, "🧬 Generating... (Live Generation)"
        except queue.Empty:
            continue

    final_result = img_queue.get()
    
    # 5. Showcase Creation
    yield final_result, None, "🖼️ Finalizing Portfolio Showcase..."
    timestamp = int(time.time())
    result_path = os.path.join(output_dir, f"result_{timestamp}.png")
    final_result.save(result_path)

    showcase_path = os.path.join(output_dir, f"showcase_{timestamp}.png")
    create_triptych(temp_input_path, temp_edge_path, result_path, showcase_path)

    # Return Final Showcase
    yield final_result, showcase_path, "✅ Generation Complete!"

# --- UI ---
theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(theme=theme, css=custom_css, title="SynthDrive Live", js=js_func) as app:
    gr.Markdown("# 🧬 Project SynthDrive")
    gr.Markdown("### Live Offline Data Factory")
    
    with gr.Row():
        # Left Column: Controls
        with gr.Column(scale=1):
            img_input = gr.Image(label="1. Input Image", height=300)
            txt_prompt = gr.Textbox(label="2. Instruction", placeholder="e.g. make it snowy", value="make it snowy")
            btn_run = gr.Button("🚀 Run Live", variant="primary")
            status_text = gr.Label(value="Ready", label="Current Status")
        
        # Right Column: Visuals
        with gr.Column(scale=2):
            # Top: The Live Movie
            img_live = gr.Image(label="🔴 Live Construction (Watch this!)", height=400)
            
            # Bottom: The Final Portfolio Piece
            img_showcase = gr.Image(label="🏆 Final Showcase", height=300)

    btn_run.click(
        fn=process_stream,
        inputs=[img_input, txt_prompt],
        outputs=[img_live, img_showcase, status_text]
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)
