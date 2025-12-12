import torch
import os
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

class SyntheticGenerator:
    def __init__(self, use_cpu=False):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "models")
        sd_path = os.path.join(model_dir, "stable-diffusion-v1-5")
        cn_path = os.path.join(model_dir, "controlnet-canny")

        if not os.path.exists(sd_path) or not os.path.exists(cn_path):
            raise FileNotFoundError("Models missing. Run setup_factory.py")

        # --- HARDWARE CONFIGURATION ---
        if use_cpu:
            self.device = "cpu"
            self.dtype = torch.float32
            print("   ... 🐢 STARTING IN CPU MODE (Safe but Slow) ...")
        else:
            self.device = "cuda"
            self.dtype = torch.float16 # FP16 is crucial for 4GB VRAM
            print("   ... 🚀 STARTING IN GPU MODE (Optimized for GTX 960M) ...")

        # 1. Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            cn_path, 
            torch_dtype=self.dtype, 
            use_safetensors=True, 
            local_files_only=True
        )

        # 2. Load Stable Diffusion
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_path, 
            controlnet=self.controlnet, 
            torch_dtype=self.dtype, 
            use_safetensors=True, 
            local_files_only=True
        )

        # 3. APPLY OPTIMIZATIONS
        # UniPC reduces steps needed (20 steps is enough)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        if self.device == "cuda":
            # --- CRITICAL FOR GTX 960M ---
            # Sequential offload saves VRAM by loading one module at a time
            self.pipe.enable_sequential_cpu_offload()
            
            # VAE Slicing prevents the "Decode" step from crashing memory
            self.pipe.enable_vae_slicing()
            
            # xformers is optional, but if installed it helps speed. 
            # We don't enforce it to avoid installation headaches.
        else:
            self.pipe.to("cpu")

    def generate(self, prompt, control_image, inference_steps=20, guidance=7.5, step_callback=None):
        n_prompt = "low quality, bad quality, sketches, cartoon, blurry, disfigured, bad anatomy"
        
        # --- REPORTER FUNCTION (Live Preview) ---
        def internal_callback(step, timestep, latents):
            # Only run if a callback was provided by the GUI
            if step_callback:
                with torch.no_grad():
                    # Decode the latent to see the image forming (Low-res preview)
                    latents = 1 / 0.18215 * latents
                    image = self.pipe.vae.decode(latents).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                    image = (image * 255).round().astype("uint8")
                    image = Image.fromarray(image[0])
                    # Send image back to the GUI
                    step_callback(image)

        output = self.pipe(
            prompt,
            negative_prompt=n_prompt,
            image=control_image,
            num_inference_steps=inference_steps,
            guidance_scale=guidance,
            callback=internal_callback, # <--- Connect the reporter
            callback_steps=1            # <--- Update every single step
        ).images[0]
        
        return output
