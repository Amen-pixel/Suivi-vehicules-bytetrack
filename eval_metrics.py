import os
import sys
import shutil

# Tentative d'importation de la bibliothèque officielle motmetrics
try:
    import motmetrics as mm
    HAS_MOTMETRICS = True
except ImportError:
    HAS_MOTMETRICS = False


def analyser_fichier_simple(fichier_log):
    """Calcule le nombre total d'IDs uniques générés (Mode sans Ground Truth)."""
    if not os.path.exists(fichier_log):
        return 0
    tous_les_ids = set()
    with open(fichier_log, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    obj_id = int(parts[1])
                    tous_les_ids.add(obj_id)
                except ValueError:
                    continue
    return len(tous_les_ids)


def sauvegarder_fichiers(dossier_drive, video_b=None, video_a=None, rapport_b="", rapport_a=""):
    """
    Crée le dossier cible sur Google Drive s'il n'existe pas,
    puis y enregistre les 2 rapports texte et y copie les 2 vidéos.
    """
    os.makedirs(dossier_drive, exist_ok=True)

    # 1. Enregistrement des rapports textes
    path_rapport_b = os.path.join(dossier_drive, "rapport_baseline.txt")
    path_rapport_a = os.path.join(dossier_drive, "rapport_hybride.txt")

    with open(path_rapport_b, "w", encoding="utf-8") as f:
        f.write(rapport_b)
    print(f"📄 Rapport Baseline sauvegardé : {path_rapport_b}")

    with open(path_rapport_a, "w", encoding="utf-8") as f:
        f.write(rapport_a)
    print(f"📄 Rapport Hybride sauvegardé  : {path_rapport_a}")

    # 2. Sauvegarde / Copie des deux vidéos
    if video_b and os.path.exists(video_b):
        dst_vid_b = os.path.join(dossier_drive, "video_baseline" + os.path.splitext(video_b)[1])
        shutil.copy(video_b, dst_vid_b)
        print(f"🎥 Vidéo Baseline copiée dans Drive : {dst_vid_b}")
    elif video_b:
        print(f"⚠️ Vidéo Baseline non trouvée sur le chemin : {video_b}")

    if video_a and os.path.exists(video_a):
        dst_vid_a = os.path.join(dossier_drive, "video_hybride" + os.path.splitext(video_a)[1])
        shutil.copy(video_a, dst_vid_a)
        print(f"🎥 Vidéo Hybride copiée dans Drive  : {dst_vid_a}")
    elif video_a:
        print(f"⚠️ Vidéo Hybride non trouvée sur le chemin : {video_a}")


def comparer_resultats(log_baseline, log_ameliore, gt_path=None, video_baseline=None, video_ameliore=None, dossier_drive="/content/drive/MyDrive/Resultats_Evaluation"):
    """
    Compare les deux modèles.
    Calcule MOTA, IDF1, IDSW si gt_path existe, sinon compare les IDs uniques.
    Génère 2 rapports .txt et sauvegarde les vidéos dans Google Drive.
    """
    entete = "=" * 60 + "\n📊 RAPPORT D'ÉVALUATION ET DE PERFORMANCE - HYBRID-BYTETRACK\n" + "=" * 60 + "\n"
    print("\n" + entete)

    rapport_baseline = entete + "--- RÉSULTATS BASELINE (ByteTrack Standard) ---\n"
    rapport_hybride = entete + "--- RÉSULTATS HYBRID-BYTETRACK ---\n"

    # CAS 1 : Ground Truth présent + motmetrics installé -> Métriques Officielles
    if gt_path and os.path.exists(gt_path) and HAS_MOTMETRICS:
        print(f"🎯 Vérité Terrain trouvée : {gt_path}")
        print("📐 Calcul des métriques officielles (MOTA, IDF1, IDSW)...\n")

        gt = mm.io.load_motchallenge(gt_path)
        mh = mm.metrics.create()

        # Evaluation Baseline
        if os.path.exists(log_baseline):
            hyp_b = mm.io.load_motchallenge(log_baseline)
            acc_b = mm.utils.compare_to_groundtruth(gt, hyp_b, 'iou', distth=0.5)
            summary_b = mh.compute(acc_b, metrics=['mota', 'idf1', 'num_switches'], name='Baseline')
            
            mota_b = f"{summary_b['mota'].iloc[0]*100:.2f}%" if summary_b['mota'].iloc[0] is not None else "-"
            idf1_b = f"{summary_b['idf1'].iloc[0]*100:.2f}%" if summary_b['idf1'].iloc[0] is not None else "-"
            idsw_b = summary_b['num_switches'].iloc[0]

            rapport_baseline += f"MOTA        : {mota_b}\n"
            rapport_baseline += f"IDF1        : {idf1_b}\n"
            rapport_baseline += f"ID Switches : {idsw_b}\n"

        # Evaluation Hybride
        if os.path.exists(log_ameliore):
            hyp_a = mm.io.load_motchallenge(log_ameliore)
            acc_a = mm.utils.compare_to_groundtruth(gt, hyp_a, 'iou', distth=0.5)
            summary_a = mh.compute(acc_a, metrics=['mota', 'idf1', 'num_switches'], name='Hybrid-ByteTrack')
            
            mota_a = f"{summary_a['mota'].iloc[0]*100:.2f}%" if summary_a['mota'].iloc[0] is not None else "-"
            idf1_a = f"{summary_a['idf1'].iloc[0]*100:.2f}%" if summary_a['idf1'].iloc[0] is not None else "-"
            idsw_a = summary_a['num_switches'].iloc[0]

            rapport_hybride += f"MOTA        : {mota_a}\n"
            rapport_hybride += f"IDF1        : {idf1_a}\n"
            rapport_hybride += f"ID Switches : {idsw_a}\n"

        # Affichage terminal global
        fichiers = [("Baseline", log_baseline), ("Hybrid-ByteTrack", log_ameliore)]
        accs, noms = [], []
        for nom, f in fichiers:
            if os.path.exists(f):
                accs.append(mm.utils.compare_to_groundtruth(gt, mm.io.load_motchallenge(f), 'iou', distth=0.5))
                noms.append(nom)
        if accs:
            summary = mh.compute_many(accs, metrics=['mota', 'idf1', 'num_switches'], names=noms, generate_overall=False)
            formatters = mh.formatters
            formatters['mota'] = lambda x: f"{x*100:.2f}%" if x is not None else "-"
            formatters['idf1'] = lambda x: f"{x*100:.2f}%" if x is not None else "-"
            print(mm.io.render_summary(summary, formatters=formatters, missing_val='-'))

    # CAS 2 : Pas de Ground Truth (ou motmetrics absent) -> Comparaison relative des IDs
    else:
        if gt_path and not HAS_MOTMETRICS:
            print("⚠️ Note: Pour voir le MOTA/IDF1, installez 'motmetrics' (pip install motmetrics).")
        
        ids_base = analyser_fichier_simple(log_baseline)
        ids_amel = analyser_fichier_simple(log_ameliore)

        print(f"🔴 Baseline (ByteTrack Standard)  : {ids_base} IDs uniques créés")
        print(f"🟢 Notre Hybrid-ByteTrack        : {ids_amel} IDs uniques créés")

        rapport_baseline += f"Nombre total d'IDs uniques générés : {ids_base}\n"
        rapport_hybride += f"Nombre total d'IDs uniques générés : {ids_amel}\n"

        diff = ids_base - ids_amel
        if ids_base > 0 and diff > 0:
            reduction = (diff / ids_base) * 100
            msg = f"\n🎉 Succès : Réduction de -{reduction:.1f}% des créations d'IDs parasites !"
            print(msg)
            rapport_hybride += f"Gain par rapport à la Baseline : -{reduction:.1f}% d'IDs parasites.\n"
        elif ids_base > 0 and diff < 0:
            print("\n⚡ Plus d'IDs détectés sur la version améliorée.")
        else:
            print("\n⚡ Stabilité similaire constatée sur cette vidéo.")

    print("=" * 60 + "\n")

    # Enregistrement final des rapports et copie des vidéos vers Google Drive
    sauvegarder_fichiers(
        dossier_drive=dossier_drive,
        video_b=video_baseline,
        video_a=video_ameliore,
        rapport_b=rapport_baseline,
        rapport_a=rapport_hybride
    )


if __name__ == "__main__":
    # Arguments de ligne de commande :
    # sys.argv[1] -> log_baseline
    # sys.argv[2] -> log_hybride
    # sys.argv[3] -> gt_path (Optionnel)
    # sys.argv[4] -> video_baseline (Optionnel)
    # sys.argv[5] -> video_hybride (Optionnel)
    # sys.argv[6] -> dossier_drive_destination (Optionnel)

    log_b = sys.argv[1] if len(sys.argv) > 1 else "results/baseline_log.txt"
    log_a = sys.argv[2] if len(sys.argv) > 2 else "results/hybride_log.txt"
    gt = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "None" else None
    
    vid_b = sys.argv[4] if len(sys.argv) > 4 else "results/baseline_video.mp4"
    vid_a = sys.argv[5] if len(sys.argv) > 5 else "results/hybride_video.mp4"
    
    drive_dir = sys.argv[6] if len(sys.argv) > 6 else "/content/drive/MyDrive/Resultats_Evaluation"

    comparer_resultats(
        log_baseline=log_b, 
        log_ameliore=log_a, 
        gt_path=gt,
        video_baseline=vid_b,
        video_ameliore=vid_a,
        dossier_drive=drive_dir
    )
