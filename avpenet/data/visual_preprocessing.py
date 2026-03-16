"""
Visual Preprocessing Pipeline for AVPENet.

Implements Algorithm 2 from the paper:
    Stage 1: Face detection (MTCNN)
    Stage 2: Landmark detection (68 points)
    Stage 3: Face alignment (rotation + crop)

Output: Aligned face image ∈ R^{224 × 224 × 3}
        Landmarks L ∈ R^{68 × 2}
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, Tuple

try:
    from facenet_pytorch import MTCNN
    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_AVAILABLE = False
    print("[WARNING] facenet-pytorch not installed. MTCNN face detection unavailable.")

try:
    import dlib
    _DLIB_AVAILABLE = True
except ImportError:
    _DLIB_AVAILABLE = False
    print("[WARNING] dlib not installed. Landmark detection unavailable.")


# ImageNet normalisation (standard torchvision preprocessing)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_SIZE   = 224
CONFIDENCE_THRESHOLD = 0.85
N_FRAMES_PER_SEGMENT = 30   # uniformly sampled from 90 frames (3 s × 30 fps)


# ─────────────────────────── Face Detection ───────────────────────────────────

class FaceDetector:
    """MTCNN-based face detector.

    Algorithm 2, Stage 1:
        {B_i} = MTCNN(I)
        B* = argmax_{B_i}(w_i × h_i)
        B_exp = expand by 20%
    """

    def __init__(self, device: str = "cpu", image_size: int = TARGET_SIZE):
        if not _MTCNN_AVAILABLE:
            raise ImportError("facenet-pytorch is required for face detection.")
        self.detector = MTCNN(
            image_size=image_size,
            margin=0,
            keep_all=True,
            device=device,
        )

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect faces and return the expanded bounding box of the largest.

        Args:
            image: BGR or RGB numpy array (H, W, 3).

        Returns:
            Expanded bounding box [x, y, w, h] or None if no face detected.
        """
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes, probs = self.detector.detect(pil_img)

        if boxes is None or len(boxes) == 0:
            return None

        # Select largest face
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        best = boxes[np.argmax(areas)]   # [x1, y1, x2, y2]

        x1, y1, x2, y2 = best
        w, h = x2 - x1, y2 - y1

        # Expand by 20%
        x1_exp = max(0, x1 - 0.1 * w)
        y1_exp = max(0, y1 - 0.1 * h)
        x2_exp = min(image.shape[1], x1_exp + 1.2 * w)
        y2_exp = min(image.shape[0], y1_exp + 1.2 * h)

        return np.array([x1_exp, y1_exp, x2_exp, y2_exp], dtype=np.float32)


# ─────────────────────────── Landmark Detection ───────────────────────────────

class LandmarkDetector:
    """dlib-based 68-point facial landmark detector.

    Algorithm 2, Stage 2.
    Requires the dlib shape predictor model file:
        shape_predictor_68_face_landmarks.dat
    """

    def __init__(self, model_path: Optional[str] = None):
        if not _DLIB_AVAILABLE:
            raise ImportError("dlib is required for landmark detection.")
        self.detector = dlib.get_frontal_face_detector()
        model_path = model_path or "shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(model_path)

    def detect(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Detect 68 landmarks within given bounding box.

        Args:
            image: BGR numpy array.
            bbox:  [x1, y1, x2, y2] bounding box.

        Returns:
            Landmarks array of shape (68, 2) or None if detection fails.
        """
        x1, y1, x2, y2 = bbox.astype(int)
        rect = dlib.rectangle(x1, y1, x2, y2)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = self.predictor(gray, rect)
        landmarks = np.array(
            [[shape.part(i).x, shape.part(i).y] for i in range(68)],
            dtype=np.float32,
        )
        return landmarks


# ─────────────────────────── Face Alignment ───────────────────────────────────

def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = TARGET_SIZE,
) -> np.ndarray:
    """Align face using eye landmarks.

    Algorithm 2, Stage 3:
        L_left  = mean(L[36:42])
        L_right = mean(L[42:48])
        θ = arctan((L_right_y - L_left_y) / (L_right_x - L_left_x))
        Apply affine rotation + crop → resize to output_size × output_size

    Args:
        image:      BGR numpy array.
        landmarks:  (68, 2) landmark coordinates.
        output_size: Target image size.

    Returns:
        Aligned RGB image of shape (output_size, output_size, 3).
    """
    # Eye centres (landmarks 36–41 = left eye, 42–47 = right eye)
    left_eye_pts  = landmarks[36:42]
    right_eye_pts = landmarks[42:48]
    left_eye_center  = left_eye_pts.mean(axis=0)
    right_eye_center = right_eye_pts.mean(axis=0)

    # Rotation angle
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotation center = midpoint between eyes
    eyes_center = (
        (left_eye_center[0] + right_eye_center[0]) / 2,
        (left_eye_center[1] + right_eye_center[1]) / 2,
    )

    # Build rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)

    # Warp image
    h, w = image.shape[:2]
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

    # Crop face region using transformed landmarks
    pts_homogeneous = np.hstack([landmarks, np.ones((68, 1))])
    new_pts = (M @ pts_homogeneous.T).T

    x1 = max(0, int(new_pts[:, 0].min()) - 10)
    y1 = max(0, int(new_pts[:, 1].min()) - 10)
    x2 = min(w, int(new_pts[:, 0].max()) + 10)
    y2 = min(h, int(new_pts[:, 1].max()) + 10)

    cropped = rotated[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = rotated   # fallback

    # Resize and convert to RGB
    aligned = cv2.resize(cropped, (output_size, output_size))
    aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned


# ─────────────────────────── Frame Preprocessing ──────────────────────────────

def preprocess_frame(
    frame: np.ndarray,
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    """Resize, convert and normalise a single BGR frame.

    Returns float32 array (3, H, W) normalised with ImageNet stats.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (target_size, target_size))
    frame_f = frame_resized.astype(np.float32) / 255.0
    frame_norm = (frame_f - IMAGENET_MEAN) / IMAGENET_STD
    return frame_norm.transpose(2, 0, 1)   # (3, H, W)


# ─────────────────────────── Full Pipeline ────────────────────────────────────

class VisualPreprocessor:
    """Full visual preprocessing pipeline (Algorithm 2 from paper).

    Processes a directory of frame images (one segment) into a
    tensor of shape (N_FRAMES, 3, 224, 224).

    Usage:
        preprocessor = VisualPreprocessor()
        frames_tensor = preprocessor(frame_dir)
    """

    def __init__(
        self,
        n_frames: int = N_FRAMES_PER_SEGMENT,
        target_size: int = TARGET_SIZE,
        use_face_detection: bool = True,
        device: str = "cpu",
    ):
        self.n_frames  = n_frames
        self.target_size = target_size
        self.use_face_detection = use_face_detection

        if use_face_detection:
            try:
                self.face_detector = FaceDetector(device=device)
            except ImportError:
                self.face_detector = None
                print("[WARNING] Face detection disabled.")
        else:
            self.face_detector = None

    def __call__(
        self,
        frame_paths: Union[list, str, Path],
    ) -> torch.Tensor:
        """
        Args:
            frame_paths: List of frame image paths, or directory path.

        Returns:
            Frame tensor of shape (N_FRAMES, 3, 224, 224).
        """
        # Collect frame paths
        if isinstance(frame_paths, (str, Path)):
            frame_dir = Path(frame_paths)
            all_paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
        else:
            all_paths = [Path(p) for p in frame_paths]

        if len(all_paths) == 0:
            # Return blank tensor
            return torch.zeros(self.n_frames, 3, self.target_size, self.target_size)

        # Uniformly sample n_frames
        indices = np.linspace(0, len(all_paths) - 1, self.n_frames, dtype=int)
        selected_paths = [all_paths[i] for i in indices]

        processed_frames = []
        for path in selected_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                frame_tensor = np.zeros((3, self.target_size, self.target_size), dtype=np.float32)
            else:
                frame_tensor = preprocess_frame(frame, self.target_size)
            processed_frames.append(frame_tensor)

        frames = np.stack(processed_frames, axis=0)   # (N, 3, H, W)
        return torch.from_numpy(frames)


# ─────────────────────────── Data Augmentation ────────────────────────────────

class VisualAugmenter:
    """Visual augmentation for training.

    Applies per-frame:
        - Random rotation (±15°)
        - Random translation (±10 px)
        - Random scaling (0.95–1.05)
        - Brightness jitter (±20%)
        - Contrast jitter (±20%)
    """

    def __init__(
        self,
        rotation_range: float = 15.0,
        translation_px: int = 10,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
        p: float = 0.5,
    ):
        self.rotation_range  = rotation_range
        self.translation_px  = translation_px
        self.scale_range     = scale_range
        self.brightness_range = brightness_range
        self.contrast_range  = contrast_range
        self.p = p

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Apply augmentations to a single BGR frame."""
        rng = np.random.default_rng()
        h, w = frame.shape[:2]

        # Geometric transform
        if rng.random() < self.p:
            angle  = rng.uniform(-self.rotation_range, self.rotation_range)
            tx     = rng.uniform(-self.translation_px, self.translation_px)
            ty     = rng.uniform(-self.translation_px, self.translation_px)
            scale  = rng.uniform(*self.scale_range)
            cx, cy = w / 2, h / 2
            M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            frame = cv2.warpAffine(frame, M, (w, h))

        # Photometric transform
        if rng.random() < self.p:
            brightness = rng.uniform(1 - self.brightness_range, 1 + self.brightness_range)
            contrast   = rng.uniform(1 - self.contrast_range, 1 + self.contrast_range)
            frame = np.clip(frame.astype(np.float32) * brightness, 0, 255)
            mean = np.mean(frame)
            frame = np.clip(contrast * (frame - mean) + mean, 0, 255)
            frame = frame.astype(np.uint8)

        return frame
