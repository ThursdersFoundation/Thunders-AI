"""Face tracking example using Thunders AI Vision.

Demonstrates face detection, face tracking setup, and face recognition
workflow. Shows how to detect faces in an image, initialize a tracker,
and identify known individuals from a face database.
"""

from thunders_ai.vision import VisionAI, VisionConfig
from thunders_ai.vision.face import FaceTracker, FaceDatabase


def main() -> None:
    """Run the face tracking and recognition example."""
    # --- Step 1: Initialize the Vision AI client ---
    config = VisionConfig(
        model="thunders-vision",
        confidence_threshold=0.7,
    )
    vision = VisionAI(config=config)
    print("Vision AI client initialized for face tracking.")
    print()

    # --- Step 2: Detect faces in an image ---
    image_path = "examples/vision/sample_images/group_photo.jpg"
    try:
        image = vision.load_image(image_path)
    except FileNotFoundError:
        image = vision.load_image("https://example.com/images/group_photo.jpg")
        print("Image loaded from URL")
    else:
        print(f"Image loaded: {image_path}")

    # Detect all faces in the image
    faces = vision.detect_faces(image)
    print(f"Detected {len(faces)} face(s) in the image.")
    for i, face in enumerate(faces):
        box = face.bbox
        print(f"  Face {i + 1}: bbox=({box[0]:.0f},{box[1]:.0f})-({box[2]:.0f},{box[3]:.0f}), "
              f"confidence={face.confidence:.1%}")
    print()

    # --- Step 3: Set up the face tracker ---
    # The tracker maintains face IDs across video frames
    tracker = FaceTracker(
        max_disappeared=30,  # Frames before a tracked face is removed
        max_distance=0.5,    # Maximum distance for face re-identification
    )
    print("Face tracker initialized.")
    print(f"  Max disappeared frames: {tracker.max_disappeared}")
    print(f"  Max re-id distance: {tracker.max_distance}")
    print()

    # --- Step 4: Register faces in the tracker ---
    # Assign persistent IDs to detected faces
    tracked_faces = tracker.update(faces)
    for tf in tracked_faces:
        print(f"  Tracked face ID={tf.track_id}, position=({tf.bbox[0]:.0f},{tf.bbox[1]:.0f})")
    print()

    # --- Step 5: Build a face database for recognition ---
    # Register known individuals with their face embeddings
    database = FaceDatabase(storage_path="/tmp/thunders_face_db")

    # Register known people (in practice, load from saved embeddings)
    database.register(name="Alice", embedding=faces[0].embedding if faces else None)
    database.register(name="Bob", embedding=faces[1].embedding if len(faces) > 1 else None)
    print(f"Face database: {len(database)} person(s) registered.")

    # --- Step 6: Recognize faces ---
    # Match detected faces against the database
    for face in faces:
        match = database.search(face.embedding, threshold=0.8)
        if match:
            print(f"  Recognized: {match.name} (distance={match.distance:.3f})")
        else:
            print(f"  Unknown face at ({face.bbox[0]:.0f},{face.bbox[1]:.0f})")
    print()

    # --- Step 7: Simulate tracking across frames ---
    # Update tracker with new detections in subsequent frames
    print("Simulating tracking across frames:")
    for frame_num in range(1, 4):
        # In practice, read from video stream here
        tracked = tracker.update(faces)
        active_ids = [tf.track_id for tf in tracked]
        print(f"  Frame {frame_num}: Active track IDs = {active_ids}")


if __name__ == "__main__":
    main()
