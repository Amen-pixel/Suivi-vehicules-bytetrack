import cv2
import numpy as np

def extraire_signature_visuelle(crop_img):
    """
    Option B: Extrait l'histogramme HSV (Teinte & Saturation) du véhicule.
    Permet une empreinte visuelle légère pour valider l'ID à la réapparition.
    """
    if crop_img is None or crop_img.size == 0:
        return None
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 25], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def comparer_signatures(hist1, hist2):
    """Calcule la corrélation d'histogramme entre deux véhicules."""
    if hist1 is None or hist2 is None:
        return 0.0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
