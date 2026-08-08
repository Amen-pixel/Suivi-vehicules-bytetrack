import cv2
import numpy as np
from ultralytics import YOLO
from src.visual_reid import extraire_signature_visuelle, comparer_signatures

class HybridTracker:
    def __init__(self, model_path="yolov8n.pt", config_path="configs/bytetrack_ameliore.yaml"):
        # Chargement du modèle YOLOv8 et du fichier de configuration ByteTrack
        self.model = YOLO(model_path)
        self.config_path = config_path
        self.signatures_vehicules = {}  # Stocke l'historique visuel des IDs : {id: signature_hsv}

    def traiter_video(self, video_path, output_txt_path, output_video_path=None, est_ameliore=True):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_idx = 0
        with open(output_txt_path, "w") as log_file:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # Choix du tracker (Baseline standard ou Config Améliorée)
                tracker_cfg = self.config_path if est_ameliore else "bytetrack.yaml"
                results = self.model.track(frame, persist=True, tracker=tracker_cfg, verbose=False)

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    cls_ids = results[0].boxes.cls.int().cpu().numpy()

                    for box, track_id, cls in zip(boxes, track_ids, cls_ids):
                        x, y, w, h = box
                        
                        # OPTION B : Extraction et mise à jour de la signature visuelle si activé
                        if est_ameliore:
                            x1, y1 = max(0, int(x - w/2)), max(0, int(y - h/2))
                            x2, y2 = min(width, int(x + w/2)), min(height, int(y + h/2))
                            crop = frame[y1:y2, x1:x2]
                            sig = extraire_signature_visuelle(crop)
                            if sig is not None:
                                self.signatures_vehicules[track_id] = sig

                        # Écriture dans le fichier log (Format MOT standard : frame, id, x, y, w, h)
                        log_file.write(f"{frame_idx},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{cls}\n")

                        # Dessin des boîtes sur la vidéo de sortie
                        if writer:
                            cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {track_id}", (int(x - w/2), int(y - h/2) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if writer:
                    writer.write(frame)

        cap.release()
        if writer:
            writer.release()
