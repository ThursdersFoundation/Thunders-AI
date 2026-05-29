"""Image classification example using Thunders AI Vision.

Demonstrates how to load an image, classify it using the VisionAI module,
and display results with confidence scores in a formatted table.
"""

from thunders_ai.vision import VisionAI, VisionConfig


def main() -> None:
    """Run the image classification example."""
    # --- Step 1: Initialize the Vision AI client ---
    # Use default configuration or customize model and confidence threshold
    config = VisionConfig(
        model="thunders-vision",
        confidence_threshold=0.5,  # Only show labels above 50% confidence
    )
    vision = VisionAI(config=config)
    print("Vision AI client initialized with model:", config.model)
    print()

    # --- Step 2: Load an image from file or URL ---
    # You can load from a local file path or a remote URL
    image_path = "examples/vision/sample_images/cat.jpg"
    image_url = "https://example.com/images/cat.jpg"

    # Try loading from file first, fall back to URL
    try:
        image = vision.load_image(image_path)
        print(f"Image loaded from file: {image_path}")
    except FileNotFoundError:
        print(f"Local file not found, loading from URL: {image_url}")
        image = vision.load_image(image_url)
    print(f"Image size: {image.width}x{image.height}")
    print()

    # --- Step 3: Classify the image ---
    # Returns top classification labels with confidence scores
    result = vision.classify(image)
    print(f"Classification result (model: {result.model})")
    print("-" * 50)

    # --- Step 4: Display results with confidence scores ---
    # Format results as a readable table
    print(f"{'Label':<30} {'Confidence':>10}")
    print(f"{'─' * 30} {'─' * 10}")
    for label in result.labels:
        # Display confidence as percentage with bar indicator
        pct = label["confidence"] * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"{label['label']:<30} {pct:>6.1f}%  {bar}")
    print()

    # Top prediction summary
    top = result.labels[0] if result.labels else None
    if top:
        print(f"Top prediction: {top['label']} ({top['confidence']:.1%} confidence)")

    # --- Step 5: Classify with a specific prompt ---
    # Use a custom prompt to guide the classification
    result = vision.classify(
        image,
        prompt="What breed of cat is this? Classify by breed.",
    )
    print("\nBreed classification:")
    for label in result.labels[:3]:
        print(f"  {label['label']}: {label['confidence']:.1%}")


if __name__ == "__main__":
    main()
