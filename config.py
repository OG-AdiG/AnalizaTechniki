"""
Konfiguracja projektu SEE Trainer - Pose Estimation
"""

# =====================================================
# KEYPOINTS - WSPÓLNY FORMAT DLA CAŁEGO ZESPOŁU
# =====================================================
# Ten plik definiuje kolejność i nazwy keypointów.
# Druga osoba w zespole (Technique Classifier) MUSI używać
# tego samego formatu i kolejności!

KEYPOINT_NAMES = [
    "nos",                    # 0
    "lewe_ucho",              # 1
    "prawe_ucho",             # 2
    "lewy_bark",              # 3  (left shoulder)
    "prawy_bark",             # 4  (right shoulder)
    "lewy_lokiec",            # 5  (left elbow)
    "prawy_lokiec",           # 6  (right elbow)
    "lewy_nadgarstek",        # 7  (left wrist)
    "prawy_nadgarstek",       # 8  (right wrist)
    "lewe_biodro",            # 9  (left hip)
    "prawe_biodro",           # 10 (right hip)
    "lewe_kolano",            # 11 (left knee)
    "prawe_kolano",           # 12 (right knee)
    "lewa_kostka",            # 13 (left ankle)
    "prawa_kostka",           # 14 (right ankle)
    "lewy_duzy_palec_stopy",  # 15 (left big toe)
    "prawy_duzy_palec_stopy", # 16 (right big toe)
    "lewa_pieta",             # 17 (left heel)
    "prawa_pieta",            # 18 (right heel)
    "mostek",                 # 19 (sternum)
    "srodek_bioder",          # 20 (mid hip)
]

NUM_KEYPOINTS = len(KEYPOINT_NAMES)  # 21

# Mapowanie COCO (17 kps) → Nasze (21 kps)
# COCO keypoints: nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder,
#                 l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip,
#                 l_knee, r_knee, l_ankle, r_ankle
COCO_TO_OURS = {
    0: 0,    # nose → nos
    3: 1,    # l_ear → lewe_ucho
    4: 2,    # r_ear → prawe_ucho
    5: 3,    # l_shoulder → lewy_bark
    6: 4,    # r_shoulder → prawy_bark
    7: 5,    # l_elbow → lewy_lokiec
    8: 6,    # r_elbow → prawy_lokiec
    9: 7,    # l_wrist → lewy_nadgarstek
    10: 8,   # r_wrist → prawy_nadgarstek
    11: 9,   # l_hip → lewe_biodro
    12: 10,  # r_hip → prawe_biodro
    13: 11,  # l_knee → lewe_kolano
    14: 12,  # r_knee → prawe_kolano
    15: 13,  # l_ankle → lewa_kostka
    16: 14,  # r_ankle → prawa_kostka
}
# Keypoints 1,2 z COCO (l_eye, r_eye) - NIE mapujemy (nie mamy oczu)
# Keypoints 15-20 nasze (palce, pięty, mostek, środek bioder) - tylko z Twoich danych

# Połączenia kości (skeleton) - do wizualizacji
SKELETON = [
    (0, 1),    # nos → lewe_ucho
    (0, 2),    # nos → prawe_ucho
    (0, 19),   # nos → mostek
    (19, 3),   # mostek → lewy_bark
    (19, 4),   # mostek → prawy_bark
    (3, 5),    # lewy_bark → lewy_lokiec
    (4, 6),    # prawy_bark → prawy_lokiec
    (5, 7),    # lewy_lokiec → lewy_nadgarstek
    (6, 8),    # prawy_lokiec → prawy_nadgarstek
    (19, 20),  # mostek → srodek_bioder
    (20, 9),   # srodek_bioder → lewe_biodro
    (20, 10),  # srodek_bioder → prawe_biodro
    (9, 11),   # lewe_biodro → lewe_kolano
    (10, 12),  # prawe_biodro → prawe_kolano
    (11, 13),  # lewe_kolano → lewa_kostka
    (12, 14),  # prawe_kolano → prawa_kostka
    (13, 15),  # lewa_kostka → lewy_duzy_palec
    (14, 16),  # prawa_kostka → prawy_duzy_palec
    (13, 17),  # lewa_kostka → lewa_pieta
    (14, 18),  # prawa_kostka → prawa_pieta
]

# =====================================================
# FORMAT WYJŚCIOWY MODELU - DLA DRUGIEJ OSOBY W ZESPOLE
# =====================================================
# Model pose estimation wypluwa:
#   keypoints: [21, 3] - dla każdego keypointa: (x, y, confidence)
#       x, y: znormalizowane 0.0 - 1.0 (procent szerokości/wysokości obrazu)
#       confidence: 0.0 - 1.0 (pewność detekcji)
#
# Osoba robiąca Technique Classifier dostaje te dane jako wejście.
# NIE dostaje obrazu - tylko 21 punktów × 3 wartości = 63 liczby na klatkę.
#
# Dla sekwencji T klatek: wejście to tensor [T, 21, 3]

OUTPUT_FORMAT = "normalized_0_1"  # x,y w zakresie 0.0 - 1.0
# NIE piksele! Normalizacja 0-1 jest lepsza bo:
# - niezależna od rozdzielczości
# - łatwiejsza do przetworzenia przez sieć neuronową
# - standard w branży

# =====================================================
# PARAMETRY TRENINGU
# =====================================================

# Rozdzielczość wejściowa modelu
INPUT_HEIGHT = 256
INPUT_WIDTH = 192

# Rozdzielczość heatmap (wyjście modelu)
HEATMAP_HEIGHT = 64
HEATMAP_WIDTH = 48

# Sigma dla Gaussian blobs w heatmapach
HEATMAP_SIGMA = 2.0

# --- Pretraining na COCO ---
COCO_TRAIN_IMAGES = "data/coco/train2017"
COCO_TRAIN_ANN = "data/coco/annotations/person_keypoints_train2017.json"
COCO_VAL_IMAGES = "data/coco/val2017"
COCO_VAL_ANN = "data/coco/annotations/person_keypoints_val2017.json"

COCO_BATCH_SIZE = 64       # Dla RTX 5070 Ti (16GB VRAM)
COCO_EPOCHS = 100
COCO_LR = 1e-3
COCO_WEIGHT_DECAY = 1e-4

# --- Fine-tuning na Twoich danych ---
CUSTOM_IMAGES = "data/custom/images"
CUSTOM_ANN = "data/custom/annotations.json"  # Format COCO JSON z CVAT

FINETUNE_BATCH_SIZE = 32
FINETUNE_EPOCHS = 80
FINETUNE_LR_BACKBONE = 1e-5   # Niska LR dla backbone (zamrożony na początku)
FINETUNE_LR_HEAD = 1e-3       # Wyższa LR dla nowego head
FINETUNE_FREEZE_EPOCHS = 20   # Ile epok z zamrożonym backbone
