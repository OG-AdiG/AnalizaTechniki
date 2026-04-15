"""
config.py — Konfiguracja projektu SEE Trainer (Technique Classifier)

Format danych wejściowych pochodzi z modelu pose estimation kolegi:
    - 21 keypointów × 3 wartości (x, y, confidence)
    - Normalizacja 0-1 (procent szerokości/wysokości obrazu)
    - Rozdzielczość wejściowa: 256×192, 30 FPS
"""
import os

# ============================================================
# ŚCIEŻKI
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_VIDEOS_DIR = os.path.join(DATA_DIR, "raw_videos")
KEYPOINTS_DIR = os.path.join(DATA_DIR, "keypoints")
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

# Ścieżka do repozytorium modelu pose estimation (SimCC)
POSE_MODEL_DIR = r"c:\Users\Adrian\Desktop\Przedsiewziecie_inzynierskie_modele_ai\pose_training"

# Ścieżka do filmów treningowych z pojedynczymi powtórzeniami
PUSHUP_VIDEOS_DIR = r"D:\Pushups"

# ============================================================
# KEYPOINTS — WSPÓLNY FORMAT ZESPOŁOWY (z config.py kolegi)
# ============================================================
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

NUM_KEYPOINTS = 21
KEYPOINT_DIMS = 3             # x, y, confidence
INPUT_CHANNELS = NUM_KEYPOINTS * KEYPOINT_DIMS  # 63

# Mapowanie nazw stawów → indeksy w naszym formacie
LANDMARK_INDEX = {
    "nose": 0,
    "left_ear": 1,   "right_ear": 2,
    "left_shoulder": 3,  "right_shoulder": 4,
    "left_elbow": 5,     "right_elbow": 6,
    "left_wrist": 7,     "right_wrist": 8,
    "left_hip": 9,       "right_hip": 10,
    "left_knee": 11,     "right_knee": 12,
    "left_ankle": 13,    "right_ankle": 14,
    "left_big_toe": 15,  "right_big_toe": 16,
    "left_heel": 17,     "right_heel": 18,
    "sternum": 19,
    "mid_hip": 20,
}

# Połączenia kości — do wizualizacji (identyczne z config kolegi)
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
    (13, 15),  # lewa_kostka → lewy_palec
    (14, 16),  # prawa_kostka → prawy_palec
    (13, 17),  # lewa_kostka → lewa_pieta
    (14, 18),  # prawa_kostka → prawa_pieta
]

# ============================================================
# PARAMETRY WIDEO I MODELU
# ============================================================
INPUT_HEIGHT = 256
INPUT_WIDTH = 192
TARGET_FPS = 30

# ============================================================
# HIPERPARAMETRY MODELU
# ============================================================
SEQUENCE_LENGTH = 30         # 30 klatek = 1s przy 30 FPS (wg dokumentu)
SEQUENCE_STRIDE = 10         # Przesunięcie sliding window
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 15

# Minimalna pewność detekcji keypointu (poniżej → punkt traktowany jako brak)
MIN_CONFIDENCE = 0.3

# ============================================================
# PARAMETRY REP COUNTER — filtr całkujący
# ============================================================
# EMA (Exponential Moving Average) — wygładzenie szumu
EMA_ALPHA = 0.25             # 0.25 = agresywne wygładzanie (mniejszy szum)

# Minimalna amplituda kąta (°) żeby policzyć powtórzenie
# Zapobiega zliczaniu mikro-ruchów / szumu keypointów
REP_AMPLITUDE_THRESHOLD = 40.0

# Minimalny czas trwania repa w klatkach (1 klatka = 1/30s)
# Fizycznie niemożliwe zrobić pompkę w < 0.67s → min 20 klatek
MIN_REP_FRAMES = 20

# Maksymalna liczba powtórzeń na minutę (per ćwiczenie)
# Zabezpieczenie: jeśli counter liczy > MAX, to sensor noise
MAX_REPS_PER_MINUTE = {
    "pushup": 40,
    "pullup_overhand": 25,
}

# Domyślny sufit jeśli ćwiczenie nie ma zdefiniowanego
DEFAULT_MAX_REPS_PER_MINUTE = 30

# ============================================================
# DEFINICJE ĆWICZEŃ — 2 ćwiczenia z klasą "setup"
# ============================================================
EXERCISE_CLASSES = {
    # -------------------------------------------------------
    # PODCIĄGANIE NACHWYTEM (PULL-UP OVERHAND)
    # -------------------------------------------------------
    "pullup_overhand": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "asymmetric_pull",
            3: "partial_rom_top",
            4: "partial_rom_bottom",
            5: "kipping",
            6: "chicken_neck",
        },
        "num_classes": 7,

        "key_landmarks": {
            "nose": 0,
            "left_ear": 1,       "right_ear": 2,
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "left_ankle": 13,    "right_ankle": 14,
            "sternum": 19,       "mid_hip": 20,
        },

        "angle_rules": {
            "elbow_angle": {
                "joints": ("shoulder", "elbow", "wrist"),
                "correct_range_top": (30, 80),       # Na górze — łokcie zgięte
                "correct_range_bottom": (160, 180),   # Na dole — ramiona proste
            },
            "shoulder_angle": {
                "joints": ("hip", "shoulder", "elbow"),
                "correct_range_top": (140, 180),
                "error_name": "łokcie zbyt szeroko/wąsko",
            },
            "body_alignment": {
                "joints": ("shoulder", "hip", "ankle"),
                "correct_range": (160, 180),
                "error_name": "kipping / kołysanie ciałem",
            },
            "neck_extension": {
                "joints": ("sternum", "nose", "left_ear"),
                "correct_range": (140, 180),
                "error_name": "naciąganie głowy (chicken neck)",
            },
        },

        "rep_phases": {
            "angle_joint": ("shoulder", "elbow", "wrist"),
            "up_threshold": 80,      # Łokcie zgięte = góra
            "down_threshold": 150,   # Łokcie proste = dół
        },
    },

    # -------------------------------------------------------
    # POMPKA (PUSH-UP)
    # Nazwy klas = nazwy folderów w D:\Pushups
    # -------------------------------------------------------
    "pushup": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "flared_elbows",
            3: "high_hips",
            4: "partial_rom_bottom",
            5: "partial_rom_top",
            6: "sagging_hips",
        },
        "num_classes": 7,

        "key_landmarks": {
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "left_ankle": 13,    "right_ankle": 14,
            "sternum": 19,       "mid_hip": 20,
        },

        "angle_rules": {
            "elbow_angle": {
                "joints": ("shoulder", "elbow", "wrist"),
                "correct_range_bottom": (70, 110),   # Na dole pompki
                "correct_range_top": (160, 180),      # Na górze pompki
            },
            "body_alignment": {
                "joints": ("shoulder", "hip", "ankle"),
                "correct_range": (160, 180),
                "error_name": "opadające/uniesione biodra",
            },
            "shoulder_angle": {
                "joints": ("hip", "shoulder", "elbow"),
                "correct_range_bottom": (30, 75),
                "error_name": "zbyt szerokie/wąskie ramiona",
            },
        },

        "rep_phases": {
            "angle_joint": ("shoulder", "elbow", "wrist"),
            "up_threshold": 150,
            "down_threshold": 100,
        },
    },
}

# Aktywne ćwiczenie (domyślnie)
ACTIVE_EXERCISE = "pushup"
NUM_CLASSES = EXERCISE_CLASSES[ACTIVE_EXERCISE]["num_classes"]
