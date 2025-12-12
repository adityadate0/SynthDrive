import argparse
import os
import sys
import time
from src.processor import ImageProcessor, AutoCaptioner
from src.generator import SyntheticGenerator

def main():
    parser = argparse.ArgumentParser(description="SynthDrive: Offline Data Factory")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--instruction", type=str, required=True, help="E.g., 'make it night', 'heavy snow'")
    parser.add_argument("--output", type=str, default="outputs/result.png")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode if GPU fails")
    args = parser.parse_args()

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    os.makedirs("outputs", exist_ok=True)

    print(f"--- 🧬  SynthDrive Engine ({'CPU' if args.cpu else 'GPU'}) ❄️  ---")

    # 1. Vision Phase (BLIP)
    try:
        # We pass the cpu flag to captioner too so it doesn't hog VRAM
        captioner = AutoCaptioner(model_dir, use_cpu=args.cpu) 
        print(f">> Analyzing Image...")
        scene_desc = captioner.generate_caption(args.input)
        print(f"   AI Saw: '{scene_desc}'")
        
        # Free up BLIP memory if on GPU, to make room for Stable Diffusion
        if not args.cpu:
            del captioner
            import torch
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ Vision Error: {e}")
        sys.exit(1)

    # 2. Prompt Engineering
    full_prompt = f"{scene_desc}, {args.instruction}, photorealistic, 8k, cinematic lighting, masterpiece"
    print(f">> Full Prompt: '{full_prompt}'")

    # 3. Geometry Phase (Canny)
    original, edges = ImageProcessor.get_canny_edges(args.input)
    # Save debug edge map
    edges.save("outputs/debug_edges.png")

    # 4. Generation Phase (SD + ControlNet)
    try:
        generator = SyntheticGenerator(use_cpu=args.cpu)
        print(">> Generating Synthetic Data...")
        start = time.time()
        
        result = generator.generate(
            prompt=full_prompt,
            control_image=edges
        )
        
        result.save(args.output)
        print(f"✅ DONE! Saved to {args.output}")
        print(f"   Time taken: {time.time() - start:.1f}s")

    except Exception as e:
        print(f"\n❌ GENERATION FAILED: {e}")
        if not args.cpu:
            print("💡 TIP: Your GPU (4GB) might be full.")
            print("   Try running again with: --cpu")

if __name__ == "__main__":
    main()
