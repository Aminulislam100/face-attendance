import cv2
import numpy as np
from pathlib import Path


# =========================================================
# FACE CASCADE
# =========================================================

def get_cascade():

    # Primary OpenCV package location
    try:
        cascade_path = (
            Path(cv2.data.haarcascades)
            / "haarcascade_frontalface_default.xml"
        )

        if cascade_path.exists():

            cascade = cv2.CascadeClassifier(
                str(cascade_path)
            )

            if not cascade.empty():
                return cascade

    except Exception:
        pass

    # Backup: look in project root
    local_path = (
        Path(__file__).parent
        / "haarcascade_frontalface_default.xml"
    )

    if local_path.exists():

        cascade = cv2.CascadeClassifier(
            str(local_path)
        )

        if not cascade.empty():
            return cascade

    raise RuntimeError(
        "Face detector could not be loaded. "
        "OpenCV Haar cascade file is missing."
    )


CASCADE = get_cascade()


# =========================================================
# IMAGE CONVERSION
# =========================================================

def rgb_to_bgr(image_rgb):

    return cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )


# =========================================================
# DETECT FACE
# =========================================================

def detect_faces(image_rgb):

    bgr = rgb_to_bgr(image_rgb)

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY
    )

    # Improve contrast
    gray = cv2.equalizeHist(gray)

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

    if len(faces) == 0:

        return None, 0

    if len(faces) > 1:

        return None, len(faces)

    x, y, w, h = faces[0]

    face = gray[
        y:y + h,
        x:x + w
    ]

    face = cv2.resize(
        face,
        size,
        interpolation=cv2.INTER_AREA
    )

    face = cv2.equalizeHist(
        face
    )

    return face, 1


# =========================================================
# FACE → JPG
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
# LBPH MODEL
# =========================================================

def build_lbph_model(samples):

    if not hasattr(
        cv2,
        "face"
    ):

        raise RuntimeError(
            "OpenCV contrib face module is unavailable."
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
# PREDICT
# =========================================================

def predict(
    model,
    face,
    threshold=75.0
):

    if model is None:

        return None, None

    label, distance = model.predict(
        face
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

def get_match_quality(
    distance
):

    if distance is None:

        return "Unknown"

    if distance <= 40:

        return "Excellent"

    if distance <= 50:

        return "Very Good"

    if distance <= 60:

        return "Good"

    if distance <= 70:

        return "Fair"

    if distance <= 75:

        return "Weak"

    return "Not recognized"
