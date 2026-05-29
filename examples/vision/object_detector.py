"""Object detection example using Thunders AI Vision.

Demonstrates how to perform object detection on an image, draw bounding
boxes around detected objects, and display detection results with class
labels, confidence scores, and bounding box coordinates.
"""

from thunders_ai.vision import VisionAI, VisionConfig


def main() -> None:
    """Run the object detection example."""
    # --- Step 1: Initialize the Vision AI client ---
    # Configure for object detection with a confidence threshold
    config = VisionConfig(
        model="thunders-vision",
        confidence_threshold=0.6,  # Only show detections above 60% confidence
    )
    vision = VisionAI(config=config)
    print("Vision AI client initialized for object detection.")
    print()

    # --- Step 2: Load the image ---
    image_path = "examples/vision/sample_images/street_scene.jpg"
    try:
        image = vision.load_image(image_path)
        print(f"Image loaded: {image_path}")
    except FileNotFoundError:
        # Use a sample URL for demonstration
        image = vision.load_image("https://example.com/images/street_scene.jpg")
        print("Image loaded from URL")
    print(f"Image dimensions: {image.width}x{image.height}")
    print()

    # --- Step 3: Run object detection ---
    # Detect all objects in the image
    detections = vision.detect_objects(image)
    print(f"Detected {len(detections)} objects:")
    print()

    # --- Step 4: Display detection results ---
    # Show each detected object with its bounding box and confidence
    print(f"{'Object':<20} {'Confidence':>10} {'Bounding Box':<30}")
    print(f"{'─' * 20} {'─' * 10} {'─' * 30}")
    for det in detections:
        box = det.bbox  # [x_min, y_min, x_max, y_max]
        box_str = f"({box[0]:.0f},{box[1]:.0f})-({box[2]:.0f},{box[3]:.0f})"
        print(f"{det.label:<20} {det.confidence:>9.1%} {box_str:<30}")
    print()

    # --- Step 5: Draw bounding boxes on the image ---
    # Annotate the image with colored bounding boxes and labels
    annotated = vision.draw_detections(image, detections)

    # Save the annotated image to disk
    output_path = "examples/vision/output/detected_objects.jpg"
    annotated.save(output_path)
    print(f"Annotated image saved to: {output_path}")

    # --- Step 6: Filter detections by class ---
    # Show only specific object categories
    target_classes = {"car", "person", "bicycle"}
    filtered = [d for d in detections if d.label in target_classes]
    print(f"\nFiltered detections ({', '.join(target_classes)}):")
    for det in filtered:
        print(f"  {det.label}: {det.confidence:.1%} at {det.bbox}")

    # --- Step 7: Count objects by category ---
    from collections import Counter
    counts = Counter(d.label for d in detections)
    print("\nObject counts by category:")
    for label, count in counts.most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
