import numpy as np
from PIL import Image
import fiftyone.zoo as foz

# 1. Load birds with segmentation masks
dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    classes=["Bird"],
    label_types=["segmentations"],
    max_samples=50
)

for i, sample in enumerate(dataset):
    # Load original image
    img = Image.open(sample.filepath).convert("RGBA")
    w, h = img.size
    
    # Get the segmentation mask for the first bird detected
    bird_label = [d for d in sample.ground_truth.detections if d.label == "Bird"][0]
    mask = bird_label.mask # This is a boolean numpy array of the object
    
    # Calculate mask bounding box in pixels
    box = bird_label.bounding_box # [x, y, w, h] normalized
    x, y, bw, bh = int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)
    
    # Resize mask to its pixel dimensions and create an Alpha layer
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((bw, bh))
    alpha = Image.new("L", (w, h), 0)
    alpha.paste(mask_img, (x, y))
    
    # Apply mask and crop to the bird's specific area
    img.putalpha(alpha)
    bird_cutout = img.crop((x, y, x + bw, y + bh))
    
    bird_cutout.save(f"./bird_assets/bird_{i}.png")