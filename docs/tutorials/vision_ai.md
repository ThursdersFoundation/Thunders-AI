# Vision AI Tutorial

Learn how to build powerful computer vision applications with Thunders AI. This tutorial covers image analysis, object detection, real-time video processing, and building a practical image classification service. Whether you're building a security camera system, a product recognition app, or an accessibility tool, Thunders AI's vision module provides the building blocks you need.

## What You'll Build

An intelligent image analysis pipeline that:
- Analyzes images for objects, scenes, and text
- Processes images from files, URLs, and cameras
- Filters detections by confidence threshold
- Handles batch processing for high-throughput workloads
- Exposes results through a simple API

## Prerequisites

- Python 3.9+
- Thunders AI with vision extras: `pip install thunders-ai[vision]`
- Sample images to test with

## Step 1: Basic Image Analysis

Start by analyzing a single image to get captions, labels, and detected objects:

```python
from thunders_ai import ThundersAI

ai = ThundersAI()

# Analyze a local image file
result = ai.analyze_image("photos/street_scene.jpg")

print("Captions:", result["captions"])
print("Labels:", result["labels"])
print("Objects:")
for obj in result["objects"]:
    print(f"  - {obj['label']}: {obj['confidence']:.1%} at {obj['bbox']}")
```

The `analyze_image()` method returns a comprehensive analysis including natural language captions, classification labels, and detected objects with bounding boxes and confidence scores.

## Step 2: Object Detection with Filtering

For applications that need precise object detection, use the `detect_objects()` method with confidence thresholds:

```python
# Detect objects with high confidence only
objects = ai.detect_objects("photos/city.jpg", confidence_threshold=0.8)

for obj in objects:
    label = obj["label"]
    confidence = obj["confidence"]
    x1, y1, x2, y2 = obj["bbox"]
    print(f"{label}: {confidence:.1%} at [{x1}, {y1}, {x2}, {y2}]")
```

Lower the threshold for recall-critical applications (e.g., safety monitoring) or raise it for precision-critical ones (e.g., product categorization).

## Step 3: Processing Images from Different Sources

Thunders AI accepts images from multiple sources, making it easy to integrate with any data pipeline:

```python
# From a URL
result = ai.analyze_image("https://example.com/product-photo.jpg")

# From base64-encoded data (useful for web APIs)
import base64
with open("photo.jpg", "rb") as f:
    b64_data = base64.b64encode(f.read()).decode()
result = ai.analyze_image(f"data:image/jpeg;base64,{b64_data}")

# From raw bytes (useful for camera frames)
with open("photo.jpg", "rb") as f:
    image_bytes = f.read()
result = ai.analyze_image(image_bytes)
```

## Step 4: Batch Processing

For processing large collections of images efficiently, use batch operations:

```python
import glob

# Process all images in a directory
image_paths = glob.glob("dataset/*.jpg")
results = ai.analyze_images(image_paths)

for path, result in zip(image_paths, results):
    print(f"{path}: {result['captions'][0]}")
    # Save results for later use
    with open(f"{path}.analysis.json", "w") as f:
        json.dump(result, f)
```

Batch processing is optimized to group inference requests, reducing overhead and improving throughput by 2-5x compared to individual calls.

## Step 5: Real-Time Video Processing

For real-time applications like security cameras or robotics, process video frames in a loop:

```python
import cv2
from thunders_ai import ThundersAI

ai = ThundersAI()
cap = cv2.VideoCapture(0)  # Open default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to bytes and analyze
    _, buffer = cv2.imencode(".jpg", frame)
    image_bytes = buffer.tobytes()

    objects = ai.detect_objects(image_bytes, confidence_threshold=0.7)

    # Draw bounding boxes on the frame
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = f"{obj['label']} {obj['confidence']:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Thunders Vision", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

## Step 6: Building a Vision API Service

Combine vision processing with the API server to create a microservice:

```python
from thunders_ai.api import serve
from thunders_ai import ThundersAI, ThundersConfig

config = ThundersConfig(model="thunders-7b-vision", device="cuda")
ai = ThundersAI(config=config)

serve(ai, host="0.0.0.0", port=8000)
```

Clients can then send images to the `/api/v1/vision/analyze` and `/api/v1/vision/detect` endpoints. See the [API Reference](../api_reference.md) for full endpoint documentation.

## Best Practices

- **Resize large images** before processing to reduce latency and memory usage
- **Use batch processing** when analyzing multiple images for better throughput
- **Set appropriate confidence thresholds** — 0.7 is a good default for most use cases
- **Cache results** for images that don't change to avoid redundant processing
- **Use GPU acceleration** for production workloads (`device="cuda"`)

## Next Steps

- Add **face detection and tracking** for people-focused applications
- Integrate with the **speech module** to create audio descriptions of images for accessibility
- Combine vision with **chat AI** for visual question answering
- Deploy on **edge devices** for low-latency real-time processing
