🚗 Hybrid-ByteTrack : Suivi Multi-Objets et Réduction des ID Switches en Trafic Dense

Python

YOLOv8

Tracker

Tracker
📌 Présentation du Projet

Dans les scénarios de trafic routier dense (comme à Cotonou), les occultations temporaires entre véhicules (motos masquées par des bus, voitures se dépassant) provoquent de fréquents sauts d'identifiants (ID Switches). L'algorithme standard ByteTrack, bien que très performant, repose uniquement sur la proximité spatiale (filtre de Kalman) et perd la trace d'un véhicule dès que l'occultation se prolonge.
Hybrid-ByteTrack est une extension optimisée de ByteTrack conçue pour maintenir des IDs uniques et stables même après des masquages prolongés.
💡 Notre Apport & Améliorations (Hybrid-ByteTrack)
Notre solution combine deux mécanismes complémentaires :
Option A — Mémoire Spatiale Étendue (track_buffer: 75) :
Augmentation du temps de rétention mémoire du filtre de Kalman de 30 frames (~1s) à 75 frames (~2.5s).
Permet de prédire la trajectoire d'un véhicule même s'il disparaît temporairement sous un masque lourd.
Option B — Ré-identification Visuelle par Signature HSV (src/visual_reid.py) :
Extraction d'une empreinte chromatique légère (histogramme HSV Teinte/Saturation) à chaque détection.
Lors de la réapparition d'un véhicule, le système compare sa signature visuelle avec les pistes enregistrées pour confirmer l'attribution de l'ID d'origine.
📁 Architecture du Dépôt

suivi-vehicules-bytetrack-hybride/
├── configs/
│   └── bytetrack_ameliore.yaml   # Configuration avancée (Buffer 75 frames)
├── src/
│   ├── visual_reid.py            # Extraction et comparaison de signatures HSV
│   └── tracker_engine.py         # Moteur de suivi YOLOv8 + Hybrid-ByteTrack
├── dataset/                      # Dossier contenant les vidéos de test (.mp4)
├── results/                      # Dossier de sortie (Vidéos annotées + Logs)
├── eval_metrics.py               # Script d'évaluation (MOTA, IDF1, ID Switches)
├── main.py                       # Pipeline principal d'exécution
├── requirements.txt              # Dépendances Python
└── README.md                     # Documentation du projet

🚀 Installation & Utilisation
1. Prérequis & Installation
Cloner le dépôt et installer les dépendances nécessaires :

git clone https://github.com/Amen-pixel/Suivi-vehicules-bytetrack
cd suivi-vehicules-bytetrack-hybride
pip install -r requirements.txt

2. Exécution du Pipeline Automatique
Placez vos vidéos de test dans le dossier dataset/, puis lancez le traitement comparatif :

python main.py


Le script va automatiquement :
Générer la vidéo et les logs de la Baseline (ByteTrack standard).
Générer la vidéo et les logs de notre Hybrid-ByteTrack.
Calculer et afficher le rapport comparatif dans la console.

3. Évaluation des Métriques
Pour évaluer les performances globales ou charger un fichier de Vérité Terrain (Ground Truth gt.txt) :

python eval_metrics.py results/video1_baseline_log.txt results/video1_ameliore_log.txt dataset/video1/gt/gt.txt

📊 Résultats & Métriques
Les métriques évaluées comprennent :
IDSW (ID Switches) : Nombre de changements/pertes d'identifiants lors d'occultations.
MOTA (Multiple Object Tracking Accuracy) : Précision globale du suivi (requiert gt.txt).
IDF1 (ID F1-Score) : Capacité du modèle à préserver l'identité exacte sur toute la séquence (requiert gt.txt).


📚 Références & Remerciements
ByteTrack : Zhang et al. (2022) — ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
Ultralytics YOLOv8 : Framework de détection d'objets en temps réel Ultralytics GitHub.




