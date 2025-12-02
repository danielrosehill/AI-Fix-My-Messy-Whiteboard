#!/usr/bin/env python3
"""
Whiteboard Processor CLI

Processes whiteboard photographs using image editing models on Replicate
to create clean, professional diagrams while preserving the original style.
"""

import argparse
import os
import sys
import time
import base64
import httpx
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table
import replicate

# Load environment variables from project root
load_dotenv(Path(__file__).parent.parent / ".env")

console = Console()

# System prompt for whiteboard cleanup
SYSTEM_PROMPT = """Transform this whiteboard photograph into a clean, professional diagram.

Create a NEW image with:
- Clean, solid MATTE WHITE background (not the original whiteboard)
- Clear, legible text preserving the original wording exactly
- Professional lines, arrows, and shapes with a hand-drawn sketch aesthetic
- Subtle colors to distinguish elements and improve clarity

Preserve the layout and spatial relationships. Use standard icons where appropriate (cylinders for databases, rectangles for systems). Fix incomplete elements. Omit board edges, reflections, erasers, and other incidental items from the photo.

The result should look like a polished whiteboard diagram ready for a professional presentation."""

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'}

# Delay between API calls (seconds) to avoid rate limiting
API_DELAY_SECONDS = 2

# Available models with their configurations
MODELS = {
    "nano-banana": {
        "id": "google/nano-banana",
        "description": "Google Gemini 2.5 Flash - fast, good quality (default)",
        "input_fn": lambda prompt, image_uri: {
            "prompt": prompt,
            "image_input": [image_uri],
            "output_format": "png"
        }
    },
    "nano-banana-pro": {
        "id": "google/nano-banana-pro",
        "description": "Google Gemini 3 Pro - highest quality, text rendering",
        "input_fn": lambda prompt, image_uri: {
            "prompt": prompt,
            "image_input": [image_uri],
            "output_format": "png"
        }
    },
    "flux-kontext-pro": {
        "id": "black-forest-labs/flux-kontext-pro",
        "description": "FLUX Kontext Pro - strong image editing",
        "input_fn": lambda prompt, image_uri: {
            "prompt": prompt,
            "input_image": image_uri,
            "aspect_ratio": "match_input_image",
            "output_format": "png"
        }
    },
    "flux-1.1-pro": {
        "id": "black-forest-labs/flux-1.1-pro",
        "description": "FLUX 1.1 Pro - composition guidance",
        "input_fn": lambda prompt, image_uri: {
            "prompt": prompt,
            "image_prompt": image_uri,
            "aspect_ratio": "1:1",
            "output_format": "png"
        }
    },
    "qwen-image-edit": {
        "id": "qwen/qwen-image-edit",
        "description": "Qwen Image Edit - precise text editing",
        "input_fn": lambda prompt, image_uri: {
            "prompt": prompt,
            "image": image_uri,
            "output_format": "png"
        }
    },
    "qwen-image-edit-plus": {
        "id": "qwen/qwen-image-edit-plus",
        "description": "Qwen Image Edit Plus - enhanced editing",
        "input_fn": lambda prompt, image_uri: {
            "prompt": prompt,
            "image": [image_uri],
            "output_format": "png"
        }
    },
}

DEFAULT_MODEL = "nano-banana"

# Directories - relative to project root (one level up from app/)
PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
QUEUE_DIR = IMAGES_DIR / "originals" / "queue"
PROCESSED_DIR = IMAGES_DIR / "originals" / "processed"
ENHANCED_DIR = IMAGES_DIR / "enhanced"


def check_api_key() -> str:
    """Check for Replicate API key and return it."""
    api_key = os.getenv("REPLICATE_API_KEY")
    if not api_key:
        console.print("[red]Error: REPLICATE_API_KEY not found in environment variables.[/red]")
        console.print("Please set it in your .env file or environment.")
        sys.exit(1)
    return api_key


def get_queue_folders() -> list[Path]:
    """Get all folders in the queue directory."""
    if not QUEUE_DIR.exists():
        console.print(f"[red]Queue directory not found: {QUEUE_DIR}[/red]")
        sys.exit(1)

    folders = [f for f in QUEUE_DIR.iterdir() if f.is_dir()]
    return sorted(folders)


def get_images_in_folder(folder: Path) -> list[Path]:
    """Get all image files in a folder."""
    images = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)
    return sorted(images)


def generate_descriptive_name(folder_name: str, index: int, total: int) -> str:
    """Generate a descriptive filename for the enhanced image."""
    timestamp = datetime.now().strftime("%Y%m%d")

    # Clean folder name for use in filename
    clean_name = folder_name.replace(" ", "-").replace("_", "-").lower()

    if total == 1:
        return f"{clean_name}-enhanced-{timestamp}.png"
    else:
        return f"{clean_name}-{index:02d}-enhanced-{timestamp}.png"


def image_to_data_uri(image_path: Path) -> str:
    """Convert an image file to a data URI."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Determine MIME type
    suffix = image_path.suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp'
    }
    mime_type = mime_types.get(suffix, 'image/png')

    # Create data URI
    b64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{b64_data}"


def get_output_url(output) -> str | None:
    """Extract URL from various Replicate output formats."""
    if output is None:
        return None

    # FileOutput object
    if hasattr(output, 'url'):
        return output.url

    # String URL
    if isinstance(output, str) and output.startswith('http'):
        return output

    # Iterator/list of outputs
    if hasattr(output, '__iter__'):
        for item in output:
            if hasattr(item, 'url'):
                return item.url
            if isinstance(item, str) and item.startswith('http'):
                return item

    # Last resort
    output_str = str(output)
    if output_str.startswith('http'):
        return output_str

    return None


def process_image(image_path: Path, output_path: Path, model_name: str) -> bool:
    """
    Process a single whiteboard image using the specified model.

    Returns True if successful, False otherwise.
    """
    try:
        model_config = MODELS[model_name]

        # Convert image to data URI for Replicate
        image_uri = image_to_data_uri(image_path)

        # Build input using model-specific function
        inputs = model_config["input_fn"](SYSTEM_PROMPT, image_uri)

        # Call model via Replicate
        output = replicate.run(model_config["id"], input=inputs)

        # Extract URL from output
        output_url = get_output_url(output)

        if output_url:
            # Download the image
            response = httpx.get(output_url, follow_redirects=True, timeout=60)
            response.raise_for_status()

            # Save the image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True

        console.print("[yellow]Warning: No image was generated in the response.[/yellow]")
        return False

    except Exception as e:
        console.print(f"[red]Error processing {image_path.name}: {e}[/red]")
        return False


def move_to_processed(folder: Path):
    """Move a processed folder from queue to processed directory."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    destination = PROCESSED_DIR / folder.name

    # Handle naming conflicts
    if destination.exists():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = PROCESSED_DIR / f"{folder.name}-{timestamp}"

    folder.rename(destination)
    return destination


def display_folder_menu(folders: list[Path]) -> Path | None:
    """Display an interactive menu to select a folder."""
    if not folders:
        console.print("[yellow]No folders found in the queue.[/yellow]")
        return None

    table = Table(title="Available Folders in Queue")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Folder Name", style="green")
    table.add_column("Images", style="yellow", justify="right")

    for i, folder in enumerate(folders, 1):
        images = get_images_in_folder(folder)
        table.add_row(str(i), folder.name, str(len(images)))

    console.print(table)
    console.print()

    while True:
        choice = Prompt.ask(
            "Select a folder number (or 'q' to quit)",
            default="1"
        )

        if choice.lower() == 'q':
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(folders):
                return folders[idx]
            else:
                console.print("[red]Invalid selection. Try again.[/red]")
        except ValueError:
            console.print("[red]Please enter a number or 'q'.[/red]")


def process_folder(folder: Path, model_name: str) -> tuple[int, int]:
    """
    Process all images in a folder.

    Returns (success_count, total_count).
    """
    images = get_images_in_folder(folder)

    if not images:
        console.print(f"[yellow]No images found in {folder.name}[/yellow]")
        return 0, 0

    # Create output directory
    output_dir = ENHANCED_DIR / folder.name
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total = len(images)

    console.print(f"\n[bold]Processing {total} image(s) from '{folder.name}' using {model_name}[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for i, image_path in enumerate(images, 1):
            task = progress.add_task(
                f"Processing {image_path.name} ({i}/{total})...",
                total=None
            )

            # Generate output filename
            output_name = generate_descriptive_name(folder.name, i, total)
            output_path = output_dir / output_name

            if process_image(image_path, output_path, model_name):
                success_count += 1
                progress.update(task, description=f"[green]Completed: {image_path.name} -> {output_name}[/green]")
            else:
                progress.update(task, description=f"[red]Failed: {image_path.name}[/red]")

            progress.remove_task(task)

            # Delay between API calls to avoid rate limiting
            if i < total:
                time.sleep(API_DELAY_SECONDS)

    return success_count, total


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transform whiteboard photos into clean, professional diagrams"
    )
    parser.add_argument(
        "-m", "--model",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Model to use for processing (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    return parser.parse_args()


def list_models():
    """Display available models."""
    table = Table(title="Available Models")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Default", style="yellow")

    for name, config in MODELS.items():
        is_default = "✓" if name == DEFAULT_MODEL else ""
        table.add_row(name, config["description"], is_default)

    console.print(table)


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    if args.list_models:
        list_models()
        return

    model_name = args.model
    model_desc = MODELS[model_name]["description"]

    console.print(Panel.fit(
        "[bold blue]Whiteboard Processor[/bold blue]\n"
        "Transform whiteboard photos into clean, professional diagrams\n"
        f"[dim]Using: {model_name} ({model_desc})[/dim]",
        border_style="blue"
    ))
    console.print()

    # Check API key
    check_api_key()

    # Get available folders
    folders = get_queue_folders()

    # Display menu and get selection
    selected_folder = display_folder_menu(folders)

    if selected_folder is None:
        console.print("[dim]Exiting...[/dim]")
        return

    # Confirm selection
    images = get_images_in_folder(selected_folder)
    console.print(f"\n[bold]Selected: {selected_folder.name}[/bold]")
    console.print(f"Contains {len(images)} image(s)")

    confirm = Prompt.ask(
        "Proceed with processing?",
        choices=["y", "n"],
        default="y"
    )

    if confirm.lower() != 'y':
        console.print("[dim]Cancelled.[/dim]")
        return

    # Process the folder
    success, total = process_folder(selected_folder, model_name)

    # Summary
    console.print()
    if success == total and total > 0:
        console.print(Panel(
            f"[green]Successfully processed all {total} image(s)![/green]\n"
            f"Enhanced images saved to: [cyan]{ENHANCED_DIR / selected_folder.name}[/cyan]",
            title="Complete",
            border_style="green"
        ))

        # Move to processed
        move_confirm = Prompt.ask(
            "Move original folder to 'processed'?",
            choices=["y", "n"],
            default="y"
        )

        if move_confirm.lower() == 'y':
            new_location = move_to_processed(selected_folder)
            console.print(f"[dim]Moved to: {new_location}[/dim]")
    elif success > 0:
        console.print(Panel(
            f"[yellow]Processed {success}/{total} image(s)[/yellow]\n"
            "Some images failed. Check the output above for details.",
            title="Partial Success",
            border_style="yellow"
        ))
    else:
        console.print(Panel(
            f"[red]Failed to process any images[/red]\n"
            "Check your API key and try again.",
            title="Failed",
            border_style="red"
        ))


if __name__ == "__main__":
    main()
