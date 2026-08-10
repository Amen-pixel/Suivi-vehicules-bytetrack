🚗 Hybrid-ByteTrack : Suivi Multi-Objets et Réduction des ID Switches

📌 Présentation du Projet
Dans les scénarios de trafic routier dense (comme à Cotonou), les occultations temporaires entre véhicules (motos masquées par des bus, voitures se dépassant) provoquent de fréquents sauts d'identifiants (ID Switches). L'algorithme standard ByteTrack, bien que très performant, repose uniquement sur la proximité spatiale (filtre de Kalman) et perd la trace d'un véhicule dès que l'occultation se prolonge.

Hybrid-ByteTrack est une extension optimisée de ByteTrack conçue pour maintenir des IDs uniques et stables même après des masquages prolongés, en s'appuyant sur un modèle de détection spécialisé et des mécanismes de mémoire avancés.

💡 Notre Apport & Pipeline d'Amélioration
Le projet suit une approche en trois étapes pour maximiser la stabilité du suivi :

1️⃣ Fine-tuning de YOLOv8 (Détection sur Mesure)
Avant toute optimisation du tracker, nous avons réalisé un fine-tuning de YOLOv8 sur un dataset spécifique au trafic local (motos, voitures, bus en environnement urbain dense).

Dataset : Images annotées représentatives du flux routier béninois.

Objectif : Améliorer le mAP (Mean Average Precision) pour réduire les faux négatifs (véhicules non détectés) qui sont la cause première des ruptures de pistes.

Résultat : Une détection plus robuste, même en cas d'occultation partielle ou de forte proximité entre les véhicules.

2️⃣ Option A — Mémoire Spatiale Étendue (track_buffer: 75)
Une fois la détection optimisée, nous agissons sur la persistance temporelle :

Amélioration : Augmentation du temps de rétention mémoire du filtre de Kalman de 30 frames (~1s) à 75 frames (~2.5s).

Impact : Permet de prédire la trajectoire d'un véhicule même s'il disparaît totalement derrière un obstacle (ex: un gros camion) pendant plus de deux secondes.

3️⃣ Option B — Ré-identification Visuelle par Signature HSV
Pour les cas où la géométrie ne suffit plus, nous ajoutons une couche de validation visuelle :

Mécanisme : Extraction d'une empreinte chromatique légère (histogramme HSV Teinte/Saturation) à chaque détection.

Impact : Lors de la réapparition d'un véhicule, le système compare sa signature visuelle avec les pistes "perdues" en mémoire pour confirmer l'attribution de l'ID d'origine, évitant ainsi la création d'un nouvel ID.

📁 Architecture du Dépôt
Plaintext
suivi-vehicules-bytetrack-hybride/
├── configs/
│   └── bytetrack_ameliore.yaml    # Configuration avancée (Buffer 75 frames)
├── src/
│   ├── visual_reid.py            # Extraction et comparaison de signatures HSV
│   └── tracker_engine.py         # Moteur de suivi YOLOv8 + Hybrid-ByteTrack
├── weights/
│   └── best_finetuned_yolo.pt    # Poids du modèle YOLOv8 après fine-tuning
├── dataset/                      # Dossier contenant les vidéos de test (.mp4)
├── results/                      # Dossier de sortie (Vidéos annotées + Logs)
├── eval_metrics.py               # Script d'évaluation (MOTA, IDF1, ID Switches)
├── main.py                       # Pipeline principal d'exécution
├── requirements.txt              # Dépendances Python
└── README.md                     # Documentation du projet
🚀 Installation & Utilisation
1. Prérequis & Installation
Bash
git clone https://github.com/Amen-pixel/Suivi-vehicules-bytetrack
cd suivi-vehicules-bytetrack-hybride
pip install -r requirements.txt
2. Exécution du Pipeline
Placez vos vidéos de test dans le dossier dataset/, puis lancez le traitement comparatif :

Bash
python main.py
Le script va utiliser votre modèle fine-tuné pour :

Générer la vidéo et les logs de la Baseline (ByteTrack standard).

Générer la vidéo et les logs de notre Hybrid-ByteTrack (Options A + B).

Afficher le rapport comparatif dans la console.

3. Évaluation des Métriques
Bash
python eval_metrics.py results/video1_baseline_log.txt results/video1_ameliore_log.txt dataset/video1/gt/gt.txt
📊 Résultats & Métriques
Nous évaluons la performance via trois indicateurs clés :

IDSW (ID Switches) : Réduction visée de plus de 40% par rapport à la baseline.

MOTA : Précision globale de détection et de suivi.

IDF1 : Capacité du modèle à maintenir l'identité sur le long terme (notre indicateur prioritaire).

📚 Références
ByteTrack : Zhang et al. (2022) — Multi-Object Tracking by Associating Every Detection Box.

YOLOv8 : Ultralytics Framework.

Dataset : Custom annotated dataset for urban traffic analysis.

Développé avec passion par Amen-pixel
