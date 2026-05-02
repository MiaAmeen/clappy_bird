import random
import os
from pathlib import Path
from PIL import Image
import numpy as np
import argparse

# Import the loader function from your existing data.py
from data import load_dataset 

class BirdAugmentor:
    def __init__(self, bird_dir, output_size=(224, 224)):
        self.bird_paths = list(Path(bird_dir).glob("*.png"))
        self.output_size = output_size
        if not self.bird_paths:
            raise FileNotFoundError("No bird PNGs found. Run harvest_birds.py first.")

    def apply_bird_occlusion(self, background_img, car_bbox, occlusion_pct):
        bg_w, bg_h = background_img.size
        
        # 1. Convert normalized car bbox to pixel area
        car_w_px = car_bbox[2] * bg_w
        car_h_px = car_bbox[3] * bg_h
        car_area_px = car_w_px * car_h_px

        # 2. Pick a bird and calculate target area
        bird_img = Image.open(random.choice(self.bird_paths)).convert("RGBA")
        target_bird_area = car_area_px * occlusion_pct

        # 3. Scale bird while maintaining aspect ratio
        bw, bh = bird_img.size
        aspect_ratio = bh / bw
        
        # Calculate dimensions
        new_bw = int(np.sqrt(target_bird_area / aspect_ratio))
        new_bh = int(new_bw * aspect_ratio)
        
        # --- SAFETY CHECK: Ensure dimensions are at least 10px ---
        # If the car is so small that the bird would be < 10px, 
        # we skip occluding this specific car.
        if new_bw < 10 or new_bh < 10:
            return background_img
        # --------------------------------------------------------

        bird_resized = bird_img.resize((new_bw, new_bh), Image.Resampling.LANCZOS)

        # 4. Determine placement
        x_min = int(car_bbox[0] * bg_w)
        y_min = int(car_bbox[1] * bg_h)
        
        # Ensure max bounds aren't smaller than min bounds
        x_max = max(x_min, int(x_min + car_w_px - new_bw))
        y_max = max(y_min, int(y_min + car_h_px - new_bh))

        paste_x = random.randint(x_min, x_max)
        paste_y = random.randint(y_min, y_max)

        # 5. Paste
        background_img.paste(bird_resized, (paste_x, paste_y), bird_resized)
        
        return background_img

if __name__ == "__main__":
    # 1. Load the dataset (This will download it if it's not there)
    # is_test=True gets TARGET_COUNT samples
    parser = argparse.ArgumentParser(description="Generate occluded car images.")
    parser.add_argument("--min_occ", type=float, default=0.05, help="Minimum occlusion (0.0-1.0)")
    parser.add_argument("--max_occ", type=float, default=0.30, help="Maximum occlusion (0.0-1.0)")
    args = parser.parse_args()
    car_dataset = load_dataset(is_test=True) 

    # 2. Setup Augmentor
    augmentor = BirdAugmentor(bird_dir="./bird_assets")
    output_path = Path("./occluded_cars_dataset_20")
    output_path.mkdir(exist_ok=True)

    # 3. Generate
    print("Generating occluded car images...")
    for i, sample in enumerate(car_dataset):
        img = Image.open(sample.filepath).convert("RGB")
        canvas = img.copy() # Create a copy to act as our drawing surface
        
        # Filter for all car detections
        car_dets = [d for d in sample.ground_truth.detections if d.label == "Car"]
        
        if not car_dets:
            continue
        
        # Loop through EVERY car found in the image
        for det in car_dets:
            # Choose a random occlusion percentage for THIS specific car
            occ_val = random.uniform(args.min_occ, args.max_occ)
            
            # Get the normalized bbox: [x, y, w, h]
            bbox = det.bounding_box
            
            # Apply the occlusion to the canvas (the current state of the image)
            # We modify the apply_bird_occlusion to return the updated canvas
            canvas = augmentor.apply_bird_occlusion(canvas, bbox, occ_val)
        
        # Final resize happens only AFTER all birds are placed
        final_img = canvas.resize(augmentor.output_size, Image.Resampling.LANCZOS)
        final_img.save(output_path / f"multi_occluded_{i:04d}.jpg")
    
    print(f"Done! Images saved to {output_path}")