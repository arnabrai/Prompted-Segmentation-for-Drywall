"""
Inference script for Drywall Prompted Segmentation.
Run on your own images to get prediction masks.

Usage:
  python predict.py --image path/to/image.jpg --prompt "segment crack"
  python predict.py --image path/to/image.jpg --prompt "segment taping area" --show
  python predict.py --image path/to/image.jpg  # runs both prompts
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "models", "checkpoints", "best_model.pt")
THRESHOLD = 0.35

ALL_PROMPTS = {
    "crack": ["segment crack", "segment wall crack"],
    "taping": ["segment taping area", "segment joint tape", "segment drywall seam"],
}


def load_model(checkpoint_path, device, zero_shot=False):
    """Load CLIPSeg model. Uses fine-tuned weights unless zero_shot=True."""
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    if not zero_shot:
        print(f"Loading fine-tuned weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✓ Fine-tuned weights loaded")
    else:
        print("⚡ Running in ZERO-SHOT mode (no fine-tuned weights)")

    model.to(device)
    model.eval()
    print(f"✓ Model ready on {device}")
    return model, processor


def predict(model, processor, image, prompt, device, threshold=THRESHOLD):
    """
    Run TTA inference (hflip, vflip, rot90) and return binary mask.
    Returns: (prob_map, binary_mask) both at original image size.
    """
    orig_w, orig_h = image.size
    img_np = np.array(image)

    versions = [
        image,
        Image.fromarray(np.fliplr(img_np)),
        Image.fromarray(np.flipud(img_np)),
        Image.fromarray(np.rot90(img_np, 1)),
    ]

    raw_probs = []
    with torch.no_grad():
        for version in versions:
            inputs = processor(
                text=[prompt], images=[version],
                return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits[0]
            probs = torch.sigmoid(logits).cpu().numpy()
            raw_probs.append(probs)

    # Undo augmentations first, then resize to original dims
    all_probs = []
    for i, probs in enumerate(raw_probs):
        if i == 1:  # hflip
            probs = np.fliplr(probs)
        elif i == 2:  # vflip
            probs = np.flipud(probs)
        elif i == 3:  # rot90 (undo rotation, then result has correct orientation)
            probs = np.rot90(probs, -1)
        # Now resize to original image dims
        probs_resized = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        all_probs.append(probs_resized)

    avg_probs = np.mean(all_probs, axis=0)
    binary_mask = (avg_probs > threshold).astype(np.uint8) * 255

    return avg_probs, binary_mask


def create_overlay(image_np, mask, color=(0, 255, 100), alpha=0.45):
    """Blend mask overlay onto original image."""
    overlay = image_np.copy()
    overlay[mask == 255] = color
    return cv2.addWeighted(image_np, 1 - alpha, overlay, alpha, 0)


def visualize(image_np, mask, prompt, prob_map, save_path=None):
    """Show side-by-side: Original | Probability Heatmap | Prediction Overlay."""
    import matplotlib.pyplot as plt

    overlay = create_overlay(image_np, mask)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Prompt: "{prompt}"', fontsize=14, fontweight="bold")

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    heatmap = axes[1].imshow(prob_map, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Probability Heatmap")
    axes[1].axis("off")
    plt.colorbar(heatmap, ax=axes[1], fraction=0.046)

    axes[2].imshow(overlay)
    pct = (mask > 0).sum() / mask.size * 100
    axes[2].set_title(f"Prediction Overlay ({pct:.1f}% covered)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Visualization saved → {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Drywall Segmentation Inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", default=None,
                        help='Text prompt, e.g. "segment crack" or "segment taping area". '
                             "If not provided, runs all prompts.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Binarization threshold (default: {THRESHOLD})")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--output-dir", default="./output",
                        help="Directory to save masks (default: ./output)")
    parser.add_argument("--show", action="store_true",
                        help="Show visualization with matplotlib")
    parser.add_argument("--device", default=None,
                        help="Device (auto-detected if not set)")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Use base CLIPSeg without fine-tuned weights")
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, processor = load_model(args.checkpoint, device, zero_shot=args.zero_shot)

    # Load image
    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    print(f"Image: {args.image} ({image.size[0]}×{image.size[1]})")

    # Determine prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        print("\nInteractive Mode: Please choose what you want to segment:")
        print("  [1] Cracks (e.g., 'segment crack')")
        print("  [2] Taping Area / Joints (e.g., 'segment taping area')")
        print("  [3] Type a custom text prompt")
        
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            prompts = ["segment crack"]
        elif choice == '2':
            prompts = ["segment taping area"]
        elif choice == '3':
            user_input = input("Enter your custom text prompt: ").strip()
            if user_input:
                prompts = [user_input]
            else:
                print("No prompt entered. Exiting.")
                sys.exit(1)
        else:
            print("Invalid choice or empty input. Exiting.")
            sys.exit(1)
            
        print(f"Got it! Segmenting: '{prompts[0]}'...")

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    print(f"\n{'='*60}")

    for prompt in prompts:
        print(f'\nPrompt: "{prompt}"')

        prob_map, mask = predict(model, processor, image, prompt, device, args.threshold)

        # Save mask
        prompt_slug = prompt.replace(" ", "_")
        mask_filename = f"{base_name}__{prompt_slug}.png"
        mask_path = os.path.join(args.output_dir, mask_filename)
        cv2.imwrite(mask_path, mask)
        print(f"  Mask saved  → {mask_path}")

        # Save overlay
        overlay = create_overlay(image_np, mask)
        overlay_filename = f"{base_name}__{prompt_slug}_overlay.png"
        overlay_path = os.path.join(args.output_dir, overlay_filename)
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  Overlay saved → {overlay_path}")

        # Stats
        coverage = (mask > 0).sum() / mask.size * 100
        print(f"  Coverage: {coverage:.2f}% of image")

        # Show
        if args.show:
            vis_path = os.path.join(args.output_dir, f"{base_name}__{prompt_slug}_vis.png")
            visualize(image_np, mask, prompt, prob_map, save_path=vis_path)

    print(f"\n{'='*60}")
    print(f"✓ All outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
