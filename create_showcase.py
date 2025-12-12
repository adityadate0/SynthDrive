import cv2
import numpy as np
import argparse
import os

def create_triptych(input_path, edge_path, output_path, save_path="showcase.png"):
    # 1. Load Images
    img_in = cv2.imread(input_path)
    img_edge = cv2.imread(edge_path)
    img_out = cv2.imread(output_path)

    if img_in is None or img_edge is None or img_out is None:
        print("❌ Error: Could not load one of the images. Check paths.")
        return

    # 2. Resize all to match (512x512)
    target_size = (512, 512)
    img_in = cv2.resize(img_in, target_size)
    img_edge = cv2.resize(img_edge, target_size)
    img_out = cv2.resize(img_out, target_size)

    # 3. Add Labels (Optional)
    # Adds a small text bar at the bottom
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_in, "Input Data", (20, 480), font, 1, (0, 255, 0), 2)
    cv2.putText(img_edge, "ControlNet Lock", (20, 480), font, 1, (255, 255, 255), 2)
    cv2.putText(img_out, "Synthetic Output", (20, 480), font, 1, (0, 255, 255), 2)

    # 4. Stitch Side-by-Side
    # Create a 20px white border between images
    border = np.ones((512, 20, 3), dtype="uint8") * 255
    final_image = np.hstack([img_in, border, img_edge, border, img_out])

    # 5. Save
    cv2.imwrite(save_path, final_image)
    print(f"✅ Portfolio Showcase saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    # We assume the edge map is always here
    edge_path = "outputs/debug_edges.png"
    
    create_triptych(args.input, edge_path, args.output)
