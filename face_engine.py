import cv2
import numpy as np
from pathlib import Path


# -------------------------------------------------
# Find Haar Cascade safely
# -------------------------------------------------

def find_cascade():

    possible_paths = []

    # OpenCV built-in cascade location
    try:
        possible_paths.append(
            Path(cv2.data.haarcascades)
            / "haarcascade_frontalface_default.xml"
        )
    except Exception:
        pass

    # Local project fallback
    possible_paths.append(
        Path(__file__).parent
        / "haarcascade_frontalface_default.xml"
    )

    for path in possible_paths:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Haar cascade file not found. "
        "Please add haarcascade_frontalface_default.xml "
        "to the project root."
    )


CASCADE_PATH = find_cascade()

CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

if CASCADE.empty():
    raise RuntimeError(
        f"Could not load Haar cascade file:\n{CASCADE_PATH}"
    )


# -------------------------------------------------
# Convert RGB → BGR
# -------------------------------------------------

def rgb_to_bgr(image_rgb):

    return cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )


# -------------------------------------------------
# Detect ONE face
# -------------------------------------------------

def prepare_face(image_rgb, size=(200, 200)):

    bgr = rgb_to_bgr(image_rgb)

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY
    )

    faces = CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None, 0

    if len(faces) > 1:
        return None, len(faces)

    x, y, w, h = faces[0]

    face = gray[
        y:y+h,
        x:x+w
    ]

    face = cv2.equalizeHist(face)

    face = cv2.resize(
        face,
        size
    )

    return face, 1


# -------------------------------------------------
# Face → JPG
# -------------------------------------------------

def face_to_jpeg(face):

    ok, encoded = cv2.imencode(
        ".jpg",
        face,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if not ok:
        raise ValueError(
            "Could not encode face image."
        )

    return encoded.tobytes()


# -------------------------------------------------
# Build LBPH model
# -------------------------------------------------

def build_lbph_model(samples):

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face module is missing. "
            "Make sure opencv-contrib-python-headless is installed."
        )

    if not samples:
        return None

    model = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
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


# -------------------------------------------------
# Predict face
# -------------------------------------------------

def predict(
    model,
    face,
    threshold=75.0
):

    if model is None:
        return None, None

    label, confidence = model.predict(face)

    if confidence > threshold:
        return None, float(confidence)

    return int(label), float(confidence)
