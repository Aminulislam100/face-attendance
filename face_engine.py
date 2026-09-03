import cv2
import numpy as np
from pathlib import Path


# =========================================================
# LOAD HAAR CASCADE
# =========================================================

def get_cascade():

    possible_paths = []

    # OpenCV built-in cascade location
    try:
        built_in_path = (
            Path(cv2.data.haarcascades)
            / "haarcascade_frontalface_default.xml"
        )

        possible_paths.append(
            built_in_path
        )

    except Exception:
        pass

    # Project root fallback
    local_path = (
        Path(__file__).parent
        / "haarcascade_frontalface_default.xml"
    )

    possible_paths.append(
        local_path
    )

    for path in possible_paths:

        if path.exists():

            cascade = cv2.CascadeClassifier(
                str(path)
            )

            if not cascade.empty():

                return cascade

    raise RuntimeError(
        "Haar cascade file could not be loaded. "
        "OpenCV installation does not contain "
        "the face detector file."
    )


CASCADE = get_cascade()


# =========================================================
# RGB TO BGR
# =========================================================

def rgb_to_bgr(image_rgb):

    return cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )


# =========================================================
# DETECT FACES
# =========================================================

def detect_faces(image_rgb):

    bgr = rgb_to_bgr(
        image_rgb
    )

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY
    )

    # Improve contrast
    gray = cv2.equalizeHist(
        gray
    )

    faces = CASCADE.detectMultiScale(
        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(80, 80)
    )

    return gray, faces


# =========================================================
# PREPARE ONE FACE
# =========================================================

def prepare_face(
    image_rgb,
    size=(200, 200)
):

    gray, faces = detect_faces(
        image_rgb
    )

    # No face
    if len(faces) == 0:

        return None, 0

    # Multiple faces
    if len(faces) > 1:

        return None, len(faces)

    x, y, w, h = faces[0]

    face = gray[
        y:y + h,
        x:x + w
    ]

    # Resize
    face = cv2.resize(
        face,
        size,
        interpolation=cv2.INTER_AREA
    )

    # Normalize contrast
    face = cv2.equalizeHist(
        face
    )

    return face, 1


# =========================================================
# FACE IMAGE TO JPG
# =========================================================

def face_to_jpeg(face):

    success, encoded = cv2.imencode(
        ".jpg",
        face,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if not success:

        raise ValueError(
            "Could not convert face to JPG."
        )

    return encoded.tobytes()


# =========================================================
# BUILD LBPH MODEL
# =========================================================

def build_lbph_model(samples):

    if not hasattr(
        cv2,
        "face"
    ):

        raise RuntimeError(
            "OpenCV Face module is unavailable. "
            "Please install opencv-contrib-python-headless."
        )

    if not samples:

        return None

    model = (
        cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
    )

    faces = [
        item[1]
        for item in samples
    ]

    labels = np.array(
        [
            item[0]
            for item in samples
        ],
        dtype=np.int32
    )

    model.train(
        faces,
        labels
    )

    return model


# =========================================================
# PREDICT FACE
# =========================================================

def predict(
    model,
    face,
    threshold=75.0
):

    if model is None:

        return None, None

    label, distance = (
        model.predict(face)
    )

    distance = float(
        distance
    )

    # IMPORTANT:
    # LBPH distance is NOT percentage.
    # Lower = generally better.
    if distance > threshold:

        return None, distance

    return (
        int(label),
        distance
    )


# =========================================================
# MATCH QUALITY
# =========================================================

def get_match_quality(distance):

    if distance is None:

        return "Unknown"

    if distance <= 40:

        return "Excellent"

    elif distance <= 50:

        return "Very Good"

    elif distance <= 60:

        return "Good"

    elif distance <= 70:

        return "Fair"

    elif distance <= 75:

        return "Weak"

    else:

        return "Not Recognized"
