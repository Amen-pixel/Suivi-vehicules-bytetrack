import cv2
import numpy as np
from ultralytics import YOLO

# Importation de tes fonctions Re-ID (Option B)
try:
    from src.visual_reid import extraire_signature_visuelle, comparer_signatures
except ImportError:
    from visual_reid import extraire_signature_visuelle, comparer_signatures


class HybridTracker:
    def __init__(self, model_path="yolov8n.pt", config_path="configs/bytetrack_ameliore.yaml"):
        # Chargement du modèle YOLOv8 et du fichier de configuration ByteTrack
        self.model = YOLO(model_path)
        self.config_path = config_path
        self.signatures_vehicules = {}  # Stocke l'historique visuel : {id: signature_hsv}

    def traiter_video(self, video_path, output_txt_path, output_video_path=None, est_ameliore=True):
        print(f"\n🚀 Traitement en cours : {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Erreur : Impossible d'ouvrir la vidéo {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # 💡 FIX 1 : Définir le tracker UNE SEULE FOIS avant la boucle (gain de vitesse)
        tracker_cfg = self.config_path if est_ameliore else "bytetrack.yaml"

        frame_idx = 0
        with open(output_txt_path, "w") as log_file:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # Tracking YOLOv8 avec le modèle fine-tuné
                results = self.model.track(frame, persist=True, tracker=tracker_cfg, verbose=False)

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    cls_ids = results[0].boxes.cls.int().cpu().numpy()

                    for box, track_id, cls in zip(boxes, track_ids, cls_ids):
                        x, y, w, h = box
                        
                        # OPTION B : Extraction de la signature visuelle HSV
                        if est_ameliore:
                            x1, y1 = max(0, int(x - w/2)), max(0, int(y - h/2))
                            x2, y2 = min(width, int(x + w/2)), min(height, int(y + h/2))
                            crop = frame[y1:y2, x1:x2]
                            
                            # 💡 FIX 2 : Vérification que l'image découpée n'est pas vide
                            if crop.size > 0:
                                sig = extraire_signature_visuelle(crop)
                                if sig is not None:
                                    self.signatures_vehicules[track_id] = sig

                        # Écriture dans le fichier log (Format MOT)
                        log_file.write(f"{frame_idx},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{cls}\n")

                        # Dessin des boîtes sur la vidéo de sortie
                        if writer:
                            # 💡 FIX 3 : Récupération et affichage "ID: X | classe"
                            class_name = self.model.names[int(cls)]
                            label = f"ID: {track_id} | {class_name}"

                            x1_draw, y1_draw = int(x - w/2), int(y - h/2)
                            x2_draw, y2_draw = int(x + w/2), int(y + h/2)

                            cv2.rectangle(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1_draw, max(20, y1_draw - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if writer:
                    writer.write(frame)

        cap.release()
        if writer:
            writer.release()

        print(f"✅ Vidéo traitée et enregistrée avec succès !")
