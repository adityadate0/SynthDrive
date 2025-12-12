import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Define paths
SD_PATH = os.path.join(MODEL_DIR, "stable-diffusion-v1-5")
CN_PATH = os.path.join(MODEL_DIR, "controlnet-canny")
BLIP_PATH = os.path.join(MODEL_DIR, "blip-image-captioning-base")

def setup():
    print("--- 🏗️  SynthDrive Factory Installer (GTX 960M Edition) 🏗️  ---")
    print(f"Target Storage: {MODEL_DIR}")
    print("NOTE: This requires Internet (~6GB download)...\n")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Download ControlNet
    if not os.path.exists(CN_PATH):
        print(">> [1/3] Downloading ControlNet (Geometry Lock)...")
        cn = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", 
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        cn.save_pretrained(CN_PATH)
        print("✅ ControlNet Installed.")
    else:
        print("✅ ControlNet already present.")

    # 2. Download Stable Diffusion
    if not os.path.exists(SD_PATH):
        print("\n>> [2/3] Downloading Stable Diffusion v1.5...")
        # Load temp pipeline to grab weights
        cn = ControlNetModel.from_pretrained(CN_PATH, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=cn,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        pipe.save_pretrained(SD_PATH)
        print("✅ Stable Diffusion Installed.")
    else:
        print("✅ Stable Diffusion already present.")

    # 3. Download BLIP (Vision)
    if not os.path.exists(BLIP_PATH):
        print("\n>> [3/3] Downloading BLIP (Vision AI)...")
        proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        proc.save_pretrained(BLIP_PATH)
        model.save_pretrained(BLIP_PATH)
        print("✅ BLIP Installed.")
    else:
        print("✅ BLIP already present.")

    print("\n" + "="*50)
    print("🎉 SETUP COMPLETE!")
    print("You may now disconnect the Internet.")
    print("="*50)

if __name__ == "__main__":
    setup()
