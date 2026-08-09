
import os
import sys

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


def comparer_resultats(log_baseline, log_ameliore, gt_path=None):
    """
    Compare les deux modèles.
    Calcule MOTA, IDF1, IDSW si gt_path existe, sinon compare les IDs uniques.
    """
    print("\n" + "=" * 60)
    print("📊 RAPPORT D'ÉVALUATION ET DE PERFORMANCE - HYBRID-BYTETRACK")
    print("=" * 60)

    # CAS 1 : Ground Truth présent + motmetrics installé -> Métriques Officielles
    if gt_path and os.path.exists(gt_path) and HAS_MOTMETRICS:
        print(f"🎯 Vérité Terrain trouvée : {gt_path}")
        print("📐 Calcul des métriques officielles (MOTA, IDF1, IDSW)...\n")

        accs = []
        noms_valides = []
        fichiers = [("Baseline", log_baseline), ("Hybrid-ByteTrack", log_ameliore)]

        for nom, fichier in fichiers:
            if os.path.exists(fichier):
                gt = mm.io.load_motchallenge(gt_path)
                hyp = mm.io.load_motchallenge(fichier)
                acc = mm.utils.compare_to_groundtruth(gt, hyp, 'iou', distth=0.5)
                accs.append(acc)
                noms_valides.append(nom)

        if accs:
            mh = mm.metrics.create()
            summary = mh.compute_many(
                accs, 
                metrics=['mota', 'idf1', 'num_switches'], 
                names=noms_valides, 
                generate_overall=False
            )

            # Formatage d'affichage propre du tableau
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

        diff = ids_base - ids_amel
        if ids_base > 0 and diff > 0:
            reduction = (diff / ids_base) * 100
            print(f"\n🎉 Succès : Réduction de -{reduction:.1f}% des créations d'IDs parasites !")
        elif ids_base > 0 and diff < 0:
            print("\n⚡ Plus d'IDs détectés sur la version améliorée.")
        else:
            print("\n⚡ Stabilité similaire constatée sur cette vidéo.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    log_b = sys.argv[1] if len(sys.argv) > 1 else "results/baseline_log.txt"
    log_a = sys.argv[2] if len(sys.argv) > 2 else "results/hybride_log.txt"
    gt = sys.argv[3] if len(sys.argv) > 3 else None

    comparer_resultats(log_b, log_a, gt)
