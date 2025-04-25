#!/usr/bin/env python3
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sam2.sam2_image_predictor import SAM2ImagePredictor

def main():
    # ─── CONFIGURATION ──────────────────────────────────────────
    # image_path = "rivendale_dataset/firefly_left/images/1739373919_100068096.png"
    image_path = "rivendale_dataset/firefly_left/images/1739373952_347697152.png"
    # box        = [600, 600, 900, 900]     # x1, y1, x2, y2 in pixel coords
    box        = [630, 740, 800, 760] 
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = None  # or "path/to/sam2_checkpoint.pth"
    config     = None  # or "path/to/sam2_config.yaml"
    # ────────────────────────────────────────────────────────────

    # 1) Build the SAM2 predictor
    if checkpoint and config:
        from sam2.build_sam import build_sam2
        sam_model = build_sam2(config, checkpoint).to(device)
        predictor = SAM2ImagePredictor(sam_model)
    else:
        predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
        predictor.model.to(device)
    predictor.model.eval()

    # 2) Load & embed the image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    predictor.set_image(img_np)

    # 3) Predict masks for our box prompt
    masks, scores, _ = predictor.predict(box=np.array(box, dtype=float))

    # 4) Pick the highest-scoring mask
    best_idx  = int(np.argmax(scores))
    best_mask = masks[best_idx]  # boolean H×W array

    # 5) Display with overlay and box
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(img_np)

    # draw the prompt box
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    rect = Rectangle((x1, y1), width, height,
                     edgecolor='red', linewidth=2, fill=False)
    ax.add_patch(rect)

    # overlay the mask
    ax.imshow(best_mask, alpha=0.5)

    ax.axis("off")
    ax.set_title("SAM2 Mask + Prompt Box Overlay")
    plt.show()

if __name__ == "__main__":
    main()
