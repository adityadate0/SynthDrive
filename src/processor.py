import cv2
import numpy as np
from PIL import Image
import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageProcessor:
    @staticmethod
    def get_canny_edges(image_path, low_threshold=100, high_threshold=200):
        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)
        
        # Canny Logic
        edges = cv2.Canny(img_array, low_threshold, high_threshold)
        edges = edges[:, :, None]
        edges = np.concatenate([edges, edges, edges], axis=2)
        
        return image, Image.fromarray(edges)

class AutoCaptioner:
    def __init__(self, model_dir, use_cpu=False):
        path = os.path.join(model_dir, "blip-image-captioning-base")
        if not os.path.exists(path):
            raise FileNotFoundError("BLIP model missing. Run setup_factory.py")
        
        # Determine Hardware
        self.device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   ... Vision Module loaded on {self.device.upper()}")

        # Load Local Models
        self.processor = BlipProcessor.from_pretrained(path, local_files_only=True)
        self.model = BlipForConditionalGeneration.from_pretrained(path, local_files_only=True).to(self.device)

    def generate_caption(self, image_path):
        raw_image = Image.open(image_path).convert('RGB')
        
        # Move inputs to the correct device (GPU or CPU)
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
        
        # Generate
        out = self.model.generate(**inputs, max_new_tokens=50)
        return self.processor.decode(out[0], skip_special_tokens=True)
