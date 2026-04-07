"""
Download ~100 car images from Open Images v6 using FiftyOne Zoo,
save them to ./data/train/, and produce a bounding-box preview grid.

FiftyOne bounding box format: [x_top_left, y_top_left, width, height]
all values normalized to [0, 1] relative to image dimensions.
"""

import random
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_COUNT = 500
OID_DIR = Path("./data/")
PREVIEW_DIR = Path("./data/previews")
PREVIEW_COLS = 5
PREVIEW_THUMB = 300
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
OID_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
DATASET_NAME_TRAIN = "open-images-v6-train"
DATASET_NAME_TEST  = "open-images-v6-test"
fo.config.dataset_zoo_dir = str(OID_DIR)

# Classes with near-zero co-occurrence with cars (indoor / nature subjects)
CLASSES = [
    "Traffic light","Traffic sign","Stop sign","Parking meter","Street light","Fire hydrant","Billboard","Tower","Skyscraper","Office building","Building","House", "Car"
]


def draw_boxes(img: Image.Image, boxes: list[list[float]]) -> Image.Image:
    """
    Draw Car bounding boxes on a copy of the image.
    FiftyOne format: [x_top_left, y_top_left, width, height] normalized [0,1].
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for (x, y, bw, bh) in boxes:
        x0, y0 = int(x * w), int(y * h)
        x1, y1 = int((x + bw) * w), int((y + bh) * h)
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    return img


def make_preview_grid(annotated: list[Image.Image], save_path: Path):
    """Tile a list of annotated images into a grid and save it as a JPEG.

    Args:
        annotated: Images already drawn with bounding boxes.
        save_path: Output path for the grid JPEG.
    """
    thumbs = [img.resize((PREVIEW_THUMB, PREVIEW_THUMB)) for img in annotated]
    rows = (len(thumbs) + PREVIEW_COLS - 1) // PREVIEW_COLS
    grid = Image.new(
        "RGB",
        (PREVIEW_COLS * PREVIEW_THUMB, rows * PREVIEW_THUMB),
        color=(30, 30, 30),
    )
    for i, thumb in enumerate(thumbs):
        grid.paste(thumb, (i % PREVIEW_COLS * PREVIEW_THUMB, i // PREVIEW_COLS * PREVIEW_THUMB))
    grid.save(save_path)
    print(f"Preview grid saved → {save_path}")


def load_dataset(is_test: bool) -> fo.Dataset:
    """Download (or reload from cache) an Open Images v6 split via FiftyOne Zoo.

    Selects CLASSES that cover urban scenes likely to contain cars, plus Car
    itself.  Prints a class-balance summary after loading.

    Args:
        is_test: If True, loads the validation split (TARGET_COUNT samples).
                 If False, loads the train split (TARGET_COUNT * 2 samples).

    Returns:
        A persistent FiftyOne Dataset with ground_truth detections attached.
    """
    max_samples = TARGET_COUNT if is_test else TARGET_COUNT * 2
    print(f"---------- LOADING DATA ----------")
    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split="test" if is_test else "train",
        label_types=["detections"],
        classes=CLASSES,
        max_samples=max_samples,
        dataset_name=DATASET_NAME_TEST if is_test else DATASET_NAME_TRAIN,
    )

    with_cars = sum(
        1 for s in dataset
        if (gt := getattr(s, "ground_truth", None)) and any(d.label == "Car" for d in gt.detections)
    )
    print(f"# CARS: {with_cars} | # NO CARS: {len(dataset) - with_cars}")
    print(f"----------------------------------")

    return dataset


def get_samples(is_test: bool) -> list[tuple[Path, int]]:
    """Return (image_path, label) for every sample. label=1 if car, 0 otherwise."""
    dataset = load_dataset(is_test=is_test)
    samples = []
    for s in dataset:
        gts = getattr(s, "ground_truth", None)
        has_car = gts is not None and any(d.label == "Car" for d in gts.detections)
        samples.append((Path(s.filepath), int(has_car)))
    return samples


def main():
    """Load the training split, extract Car bounding boxes, and write a
    bounding-box preview grid to PREVIEW_DIR."""
    dataset = load_dataset(is_test=False)

    print("Extracting Car bounding boxes ...")
    samples_info: list[tuple[Path, list]] = []
    for sample in dataset:
        car_boxes = []
        detections = getattr(sample, "ground_truth", None)
        if detections:
            for det in detections.detections:
                if det.label == "Car":
                    car_boxes.append(det.bounding_box)  # [x, y, w, h] normalized
        samples_info.append((Path(sample.filepath), car_boxes))

    # 3. Draw bounding boxes and build a preview grid
    print("Generating bounding-box preview ...")
    preview_n = min(25, len(samples_info))
    preview_sample = random.sample(samples_info, preview_n)

    annotated_imgs = []
    for path, boxes in preview_sample:
        img = Image.open(path).convert("RGB")
        annotated_imgs.append(draw_boxes(img, boxes))

    make_preview_grid(annotated_imgs, PREVIEW_DIR / "car_bbox_preview.jpg")
    print("Done.")


if __name__ == "__main__":
    main()
