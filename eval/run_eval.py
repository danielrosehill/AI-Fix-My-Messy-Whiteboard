#!/usr/bin/env python3
"""
Whiteboard Enhancement Model Evaluation

Runs test images through multiple Replicate models to compare results.
"""

import os
import sys
import base64
import httpx
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import replicate

# Load environment variables from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Directories
EVAL_DIR = Path(__file__).parent
SOURCE_DIR = EVAL_DIR / "source"
RUNS_DIR = EVAL_DIR / "runs"

# The prompt to use for all models
PROMPT = """Transform this whiteboard photograph into a clean, professional diagram.

Create a NEW image with:
- Clean, solid MATTE WHITE background (not the original whiteboard)
- Clear, legible text preserving the original wording exactly
- Professional lines, arrows, and shapes with a hand-drawn sketch aesthetic
- Subtle colors to distinguish elements and improve clarity

Preserve the layout and spatial relationships. Use standard icons where appropriate (cylinders for databases, rectangles for systems). Fix incomplete elements. Omit board edges, reflections, erasers, and other incidental items from the photo.

The result should look like a polished whiteboard diagram ready for a professional presentation."""

# Models to evaluate with their specific configurations
MODELS = {
    "flux-kontext-pro": {
        "id": "black-forest-labs/flux-kontext-pro",
        "input_fn": lambda prompt, image_uri: {
            "prompt": prompt,
            "input_image": image_uri,
            "aspect_ratio": "match_input_image",
            "output_format": "png"
        }
    },
    "flux-1.1-pro": {
        "id": "black-forest-labs/flux-1.1-pro",
        "input_fn": lambda prompt, image_uri: {
            "prompt": prompt,
            "image_prompt": image_uri,  # Note: This is for composition guidance, not direct editing
            "aspect_ratio": "1:1",
            "output_format": "png"
        }
    },
    "qwen-image-edit": {
        "id": "qwen/qwen-image-edit",
        "input_fn": lambda prompt, image_uri: {
            "prompt": prompt,
            "image": image_uri,  # Single string, returns array of URIs
            "output_format": "png"
        }
    },
    "qwen-image-edit-plus": {
        "id": "qwen/qwen-image-edit-plus",
        "input_fn": lambda prompt, image_uri: {
            "prompt": prompt,
            "image": [image_uri],  # Expects array
            "output_format": "png"
        }
    },
    "nano-banana": {
        "id": "google/nano-banana",
        "input_fn": lambda prompt, image_uri: {
            "prompt": prompt,
            "image_input": [image_uri],  # Expects array
            "output_format": "png"
        }
    },
    "nano-banana-pro": {
        "id": "google/nano-banana-pro",
        "input_fn": lambda prompt, image_uri: {
            "prompt": prompt,
            "image_input": [image_uri],  # Expects array
            "output_format": "png"
        }
    }
}


def check_api_key() -> str:
    """Check for Replicate API key and return it."""
    api_key = os.getenv("REPLICATE_API_KEY")
    if not api_key:
        print("Error: REPLICATE_API_KEY not found in environment variables.")
        sys.exit(1)
    return api_key


def image_to_data_uri(image_path: Path) -> str:
    """Convert an image file to a data URI."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    suffix = image_path.suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
    }
    mime_type = mime_types.get(suffix, 'image/png')

    b64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{b64_data}"


def run_model(model_name: str, model_config: dict, image_uri: str) -> str | None:
    """Run a single model and return the output URL."""
    try:
        inputs = model_config["input_fn"](PROMPT, image_uri)
        output = replicate.run(model_config["id"], input=inputs)

        # Handle different output formats
        if output is None:
            return None

        # FileOutput object - has a url attribute or can be converted to string
        if hasattr(output, 'url'):
            return output.url

        # String URL
        if isinstance(output, str):
            if output.startswith('http'):
                return output
            return None

        # Iterator/list of outputs
        if hasattr(output, '__iter__'):
            for item in output:
                if hasattr(item, 'url'):
                    return item.url
                if isinstance(item, str) and item.startswith('http'):
                    return item

        # Last resort: try converting to string
        output_str = str(output)
        if output_str.startswith('http'):
            return output_str

        return None
    except Exception as e:
        print(f"  Error with {model_name}: {e}")
        return None


def download_image(url: str, output_path: Path) -> bool:
    """Download an image from URL to local path."""
    try:
        response = httpx.get(url, follow_redirects=True, timeout=60)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"  Download error: {e}")
        return False


def main():
    print("=" * 60)
    print("Whiteboard Enhancement Model Evaluation")
    print("=" * 60)

    check_api_key()

    # Get source images
    source_images = sorted(SOURCE_DIR.glob("*.png"))
    if not source_images:
        print(f"No PNG images found in {SOURCE_DIR}")
        sys.exit(1)

    print(f"\nFound {len(source_images)} source image(s)")
    print(f"Testing {len(MODELS)} model(s)")

    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {run_dir}\n")

    # Process each image with each model
    results = []

    for img_path in source_images:
        print(f"\nProcessing: {img_path.name}")
        print("-" * 40)

        image_uri = image_to_data_uri(img_path)
        img_stem = img_path.stem

        for model_name, model_config in MODELS.items():
            print(f"  Running {model_name}...", end=" ", flush=True)

            output_url = run_model(model_name, model_config, image_uri)

            if output_url:
                output_path = run_dir / f"{img_stem}_{model_name}.png"
                if download_image(output_url, output_path):
                    print(f"✓ Saved")
                    results.append((img_path.name, model_name, "success"))
                else:
                    print(f"✗ Download failed")
                    results.append((img_path.name, model_name, "download_failed"))
            else:
                print(f"✗ Failed")
                results.append((img_path.name, model_name, "api_failed"))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successes = sum(1 for r in results if r[2] == "success")
    total = len(results)

    print(f"Total: {successes}/{total} successful")
    print(f"Results saved to: {run_dir}")

    # Save results log
    log_path = run_dir / "results.txt"
    with open(log_path, 'w') as f:
        f.write(f"Evaluation Run: {timestamp}\n")
        f.write(f"Prompt:\n{PROMPT}\n\n")
        f.write("Results:\n")
        for img, model, status in results:
            f.write(f"  {img} + {model}: {status}\n")

    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
