import cv2
import numpy as np

def enhance_frame_clahe(frame):
    """Enhances input frame using CLAHE (Contrast Limited Adaptive 
    Histogram Equalization)."""

    # convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)
    return enhanced

def enhance_frame_gamma(frame, gamma=1.5):
    """Enhances input frame using gamma correction for brightness adjustment."""

    # Build lookup table
    table = np.array([((i/255.0) ** (1.0/gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(frame, table)