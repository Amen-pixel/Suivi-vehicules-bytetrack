import os
from src.tracker_engine import HybridTracker
from eval_metrics import comparer_resultats

def executer_pipeline():
    os.makedirs("results", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    # Récupère la liste des vidéos dans le dossier dataset
    videos = [f for f in os.listdir("dataset") if f.endswith((".mp4", ".avi", ".mov"))]

    if not videos:
        print("⚠️ Aucune vidéo trouvée dans le dossier 'dataset/'. Veuillez en ajouter une.")
        return

    # Utilisation prioritaire du modèle fine-tuné s'il existe
    model_path = "experiments/yolov8_benin/weights/best.pt"
    if not os.path.exists(model_path):
        print("⚠️ Poids fine-tunés introuvables. Utilisation du modèle 'yolov8n.pt' par défaut.")
        model_path = "yolov8n.pt"
    else:
        print(f"🎯 Chargement du modèle spécialisé : {model_path}")

    tracker = HybridTracker(model_path=model_path)

    for vid in videos:
        input_path = os.path.join("dataset", vid)
        nom_base = os.path.splitext(vid)[0]

        print(f"\n🎬 Traitement de la vidéo : {vid}")

        # 1. Passage Baseline
        print("  ⏳ [1/2] Exécution de ByteTrack Standard (Baseline)...")
        log_base = f"results/{nom_base}_baseline_log.txt"
        vid_base = f"results/{nom_base}_baseline_out.mp4"
        tracker.traiter_video(input_path, log_base, vid_base, est_ameliore=False)

        # 2. Passage Modèle Amélioré (Hybrid)
        print("  ⏳ [2/2] Exécution de Hybrid-ByteTrack (Votre Solution)...")
        log_amel = f"results/{nom_base}_ameliore_log.txt"
        vid_amel = f"results/{nom_base}_ameliore_out.mp4"
        tracker.traiter_video(input_path, log_amel, vid_amel, est_ameliore=True)

        # 3. Affichage des métriques et sauvegarde des vidéos sur Drive
        comparer_resultats(
            log_base, 
            log_amel, 
            gt_path=None, 
            video_baseline=vid_base, 
            video_ameliore=vid_amel
        )

if __name__ == "__main__":
    executer_pipeline()
