import cv2
import numpy as np


CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def rgb_to_bgr(image_rgb):
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def prepare_face(image_rgb, size=(200, 200)):
    """
    Detect exactly one face.
    Returns (gray_face, face_count).
    """
    bgr = rgb_to_bgr(image_rgb)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    faces = CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return None, 0

    if len(faces) > 1:
        return None, len(faces)

    x, y, w, h = faces[0]

    face = gray[y:y+h, x:x+w]
    face = cv2.equalizeHist(face)
    face = cv2.resize(face, size)

    return face, 1


def face_to_jpeg(face):
    ok, encoded = cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Could not encode face image.")
    return encoded.tobytes()


def build_lbph_model(samples):
    """
    samples = [(label_int, gray_face_ndarray), ...]
    """
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
        grid_y=8,
    )

    faces = [item[1] for item in samples]
    labels = np.array([item[0] for item in samples], dtype=np.int32)

    model.train(faces, labels)
    return model


def predict(model, face, threshold=75.0):
    """
    LBPH: lower confidence/distance is generally better.
    Returns label or None.
    """
    if model is None:
        return None, None

    label, confidence = model.predict(face)

    if confidence > threshold:
        return None, float(confidence)

    return int(label), float(confidence)
