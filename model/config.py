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
# Konfigurowalna przez zmienną środowiskową POSE_MODEL_DIR,
# domyślnie ../pose_training (side-by-side z tym repo).
POSE_MODEL_DIR = os.environ.get(
    "POSE_MODEL_DIR",
    os.path.join(os.path.dirname(PROJECT_ROOT), "pose_training"),
)

# Ścieżka do filmów treningowych z pojedynczymi powtórzeniami
# Konfigurowalna przez zmienną środowiskową PUSHUP_VIDEOS_DIR.
PUSHUP_VIDEOS_DIR = os.environ.get(
    "PUSHUP_VIDEOS_DIR",
    os.path.join(DATA_DIR, "raw_videos", "pushup"),
)

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
DROPOUT_RATE = 0.5
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
MIN_REP_FRAMES = 10  # 10 klatek @ 30 FPS ≈ 0.34s min między repami

# Maksymalna liczba powtórzeń na minutę (per ćwiczenie)
# Zabezpieczenie: jeśli counter liczy > MAX, to sensor noise
MAX_REPS_PER_MINUTE = {
    "pushup": 40,
    "pullup": 25,
    "dips": 35,
    "leg_raise": 25,
    "bench_press": 25,
    "deadlift": 20,
    "squat": 35,
    "bicep_curl": 35,
    "v_sit": 30,
    "lateral_raise": 35,
    "ohp": 25,
    "bent_over_row": 30,
    "french_press": 30,
    "chest_fly": 25,
    "muscle_up": 15,
    "lunge": 30,
    "hip_thrust": 30,
    "v_up": 30,
}

# Domyślny sufit jeśli ćwiczenie nie ma zdefiniowanego
DEFAULT_MAX_REPS_PER_MINUTE = 30

# --- Adaptacyjne progi (NOWE) ---
# Po ilu repach dostosować progi do faktycznego ROM osoby
ADAPTIVE_THRESHOLD_REPS = 3
# Margines od obserwowanego ROM (0.2 = 20% zapasu)
ADAPTIVE_MARGIN = 0.20

# --- Odporność na dropout keypointów (NOWE) ---
# Max klatek z rzędu bez ważnego kąta — potem sygnał uznany za utracony
MAX_CONSECUTIVE_DROPOUTS = 5

# --- Timeout fazy (NOWE) ---
# Max klatek w jednej fazie (np. "down") zanim wymusimy granicę repa
# 150 klatek = 5 sekund przy 30 FPS
MAX_PHASE_FRAMES = 150

# --- Detekcja partial repów (pół-powtórzeń) ---
# Min. amplituda (°) żeby uznać ruch za partial rep (niższa niż full rep)
PARTIAL_REP_MIN_AMPLITUDE = 20.0
# Deadband pochodnej (°/klatkę) — poniżej tego szum, nie zmiana kierunku
REVERSAL_DEADBAND = 0.5

# ============================================================
# DEFINICJE ĆWICZEŃ
# ============================================================
# Aktywnie trenowane (14 ćwiczeń z listy SEE):
#   pullup, dips, pushup, bench_press, deadlift, bicep_curl,
#   lateral_raise, ohp, bent_over_row, french_press, chest_fly,
#   l_sit, hip_thrust, v_sit
# Zaplanowane (struktura bez labeli/angle_rules):
#   leg_raise, squat, muscle_up, lunge, plank, hollow_body
#
# Każde ćwiczenie ma klasę 0 = "setup" (pozycja przygotowawcza,
# między-rep przerwa). Klasyfikator filtruje setup zanim policzy
# powtórzenie — patrz model/rep_classifier.py.
# ============================================================
EXERCISE_CLASSES = {
    # -------------------------------------------------------
    # 1. PODCIĄGANIE (PULL-UP)
    # -------------------------------------------------------
    "pullup": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "kipping",
            3: "half_rep_top",
            4: "half_rep_bottom",
            5: "chicken_neck",
            6: "asymmetric",
        },
        "num_classes": 7,
        "key_landmarks": {
            "nose": 0, "left_ear": 1, "right_ear": 2,
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
                "correct_range_top": (30, 80),
                "correct_range_bottom": (160, 180),
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
            "up_threshold": 150,
            "down_threshold": 80,
        },
    },

    # -------------------------------------------------------
    # 2. DIPY
    # -------------------------------------------------------
    "dips": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "arching_back",
            3: "flared_elbows",
            4: "half_rep_bottom",
            5: "half_rep_top",
            6: "kipping",
            7: "shoulders_not_retracted",
        },
        "num_classes": 8,
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
                "correct_range_bottom": (70, 90),
                "correct_range_top": (160, 180),
            },
            "body_alignment": {
                "joints": ("shoulder", "hip", "ankle"),
                "correct_range": (150, 180),
                "error_name": "kipping / bujanie",
            },
            "shoulder_angle": {
                "joints": ("hip", "shoulder", "elbow"),
                "correct_range_bottom": (20, 60),
                "error_name": "flared elbows (łokcie na zewnątrz)",
            },
        },
        "rep_phases": {
            "angle_joint": ("shoulder", "elbow", "wrist"),
            "up_threshold": 140,
            "down_threshold": 100,
        },
    },

    # -------------------------------------------------------
    # 3. POMPKI (PUSH-UP)
    # -------------------------------------------------------
    "pushup": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "flared_elbows",
            3: "half_rep_top",
            4: "half_rep_bottom",
            5: "high_hips",
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
                "correct_range_bottom": (70, 110),
                "correct_range_top": (160, 180),
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
            "up_threshold": 125,
            "down_threshold": 100,
        },
    },

    # -------------------------------------------------------
    # 4. UNOSZENIE NÓG (LEG RAISE)
    # -------------------------------------------------------
    "leg_raise": {
        "labels": {
            0: "setup",
            1: "arching_back",
            2: "bent_arms",
            3: "correct",
            4: "half_rep_bottom",
            5: "half_rep_top",
            6: "no_retraction",
        },
        "num_classes": 7,
        "key_landmarks": {
            "nose": 0, "left_ear": 1, "right_ear": 2,
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "left_ankle": 13,    "right_ankle": 14,
            "left_big_toe": 15,  "right_big_toe": 16,
            "left_heel": 17,     "right_heel": 18,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "hip_angle": {
                "joints": ("shoulder", "hip", "knee"),
                "correct_range_top": (60, 100),
                "correct_range_bottom": (160, 180),
                "error_name": "niepełny zakres ruchu (hip)",
            },
            "elbow_angle": {
                "joints": ("shoulder", "elbow", "wrist"),
                "correct_range": (160, 180),
                "error_name": "zgięte ręce (bent_arms)",
            },
            "back_alignment": {
                "joints": ("sternum", "mid_hip", "shoulder"),
                "correct_range": (160, 180),
                "error_name": "wygięte plecy (arching_back)",
            },
        },
        "rep_phases": {
            "angle_joint": ("shoulder", "hip", "knee"),
            "up_threshold": 150,
            "down_threshold": 100,
        },
    },

    # -------------------------------------------------------
    # 5. WYCISKANIE NA KLATĘ (BENCH PRESS)
    # -------------------------------------------------------
    "bench_press": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "asymmetric",
            3: "flared_elbows",
            4: "no_arch",
            5: "raised_hips",
        },
        "num_classes": 6,
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
                "correct_range_bottom": (70, 100),
                "correct_range_top": (160, 180),
                "error_name": "niepełny zakres ruchu (łokieć)",
            },
            "shoulder_angle": {
                "joints": ("hip", "shoulder", "elbow"),
                "correct_range_bottom": (30, 75),
                "error_name": "flared elbows (łokcie zbyt szeroko)",
            },
            "back_arch": {
                "joints": ("shoulder", "mid_hip", "knee"),
                "correct_range": (155, 180),
                "error_name": "brak mostkowania (no_arch)",
            },
            "hip_contact": {
                "joints": ("shoulder", "hip", "knee"),
                "correct_range": (140, 180),
                "error_name": "uniesione biodra (raised_hips)",
            },
        },
        "rep_phases": {
            "angle_joint": ("shoulder", "elbow", "wrist"),
            "up_threshold": 155,
            "down_threshold": 105,
        },
    },

    # -------------------------------------------------------
    # 6. MARTWY CIĄG (DEADLIFT)
    # -------------------------------------------------------
    "deadlift": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "asymmetric",
            3: "shoulders_not_retracted",
            4: "hips_same_as_knees",
            5: "arching_back",
        },
        "num_classes": 6,
        "key_landmarks": {
            "nose": 0, "left_ear": 1, "right_ear": 2,
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "left_ankle": 13,    "right_ankle": 14,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "hip_hinge": {
                "joints": ("shoulder", "hip", "knee"),
                "correct_range_bottom": (60, 100),
                "correct_range_top": (160, 180),
                "error_name": "niepełny zakres ruchu (hip hinge)",
            },
            "knee_angle": {
                "joints": ("hip", "knee", "ankle"),
                "correct_range_bottom": (100, 140),
                "correct_range_top": (160, 180),
                "error_name": "zbyt zgięte/proste kolana",
            },
            "back_alignment": {
                "joints": ("sternum", "mid_hip", "knee"),
                "correct_range": (140, 180),
                "error_name": "zaokrąglone plecy (banana back)",
            },
        },
        "rep_phases": {
            "angle_joint": ("shoulder", "hip", "knee"),
            "up_threshold": 150,
            "down_threshold": 110,
        },
    },

    # -------------------------------------------------------
    # 7. PRZYSIADY (SQUAT)
    # -------------------------------------------------------
    "squat": {
        "labels": {
            0: "setup",
            1: "arching_back",
            2: "correct",
            3: "half_rep_bottom",
            4: "half_rep_top",
            5: "inward_knee",
        },
        "num_classes": 6,
        "key_landmarks": {
            "nose": 0, "left_ear": 1, "right_ear": 2,
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "left_ankle": 13,    "right_ankle": 14,
            "left_big_toe": 15,  "right_big_toe": 16,
            "left_heel": 17,     "right_heel": 18,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "knee_angle": {
                "joints": ("hip", "knee", "ankle"),
                "correct_range_bottom": (70, 100),
                "correct_range_top": (160, 180),
            },
            "back_alignment": {
                "joints": ("shoulder", "hip", "knee"),
                "correct_range": (40, 90),
                "error_name": "garb / wygięte plecy (arching_back)",
            },
        },
        "rep_phases": {
            "angle_joint": ("hip", "knee", "ankle"),
            "up_threshold": 150,
            "down_threshold": 95,
        },
    },

    # -------------------------------------------------------
    # 8. UGINANIE NA BICEPS (BICEP CURL)
    # -------------------------------------------------------
    "bicep_curl": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "half_rep_top",
            3: "half_rep_bottom",
            4: "asymmetric",
            5: "swinging",
            6: "flared_elbows",
        },
        "num_classes": 7,
        "key_landmarks": {
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "elbow_angle": {
                "joints": ("shoulder", "elbow", "wrist"),
                "correct_range_top": (30, 60),
                "correct_range_bottom": (150, 180),
                "error_name": "niepełny zakres ruchu (łokieć)",
            },
            "body_alignment": {
                "joints": ("shoulder", "hip", "knee"),
                "correct_range": (160, 180),
                "error_name": "kipping / kołysanie ciałem",
            },
            "elbow_flare": {
                "joints": ("hip", "shoulder", "elbow"),
                "correct_range": (0, 20),
                "error_name": "łokcie odchylone od tułowia",
            },
        },
        "rep_phases": {
            "angle_joint": ("shoulder", "elbow", "wrist"),
            "up_threshold": 140,
            "down_threshold": 80,
        },
    },

    # -------------------------------------------------------
    # 9. SCYZORYKI (V-SIT)
    # -------------------------------------------------------
    "v_sit": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "arching_back",
            3: "dropping_legs",
            4: "half_rep_top",
            5: "half_rep_bottom",
        },
        "num_classes": 6,
        "key_landmarks": {
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "left_ankle": 13,    "right_ankle": 14,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "hip_angle": {
                "joints": ("shoulder", "hip", "knee"),
                "correct_range_bottom": (60, 100),
                "correct_range_top": (160, 180),
                "error_name": "niepełny zakres ruchu (V-sit)",
            },
            "knee_angle": {
                "joints": ("hip", "knee", "ankle"),
                "correct_range": (155, 180),
                "error_name": "zgięte nogi",
            },
            "back_alignment": {
                "joints": ("sternum", "mid_hip", "knee"),
                "correct_range": (150, 180),
                "error_name": "wygięte plecy (arching_back)",
            },
        },
        "rep_phases": {
            "angle_joint": ("shoulder", "hip", "knee"),
            "up_threshold": 150,
            "down_threshold": 95,
        },
    },

    # -------------------------------------------------------
    # 10. UNOSZENIE HANTLI (LATERAL RAISE)
    # -------------------------------------------------------
    # UWAGA: użytkownik nie wymienił "setup" w swojej liście, ale klasa
    # setup jest niezbędna w przepływie (rep_classifier filtruje setup
    # zanim policzy powtórzenie). Dodano klasę 0 = setup automatycznie.
    "lateral_raise": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "bent_arms",
            3: "half_rep_bottom",
            4: "half_rep_top",
            5: "straight_arms",
        },
        "num_classes": 6,
        "key_landmarks": {
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "shoulder_abduction": {
                "joints": ("hip", "shoulder", "elbow"),
                "correct_range_top": (75, 100),
                "correct_range_bottom": (5, 25),
                "error_name": "niepełny zakres ruchu (lateral raise)",
            },
            "elbow_angle": {
                "joints": ("shoulder", "elbow", "wrist"),
                "correct_range": (150, 175),
                "error_name": "zbyt zgięte/proste ręce",
            },
        },
        "rep_phases": {
            "angle_joint": ("hip", "shoulder", "elbow"),
            "up_threshold": 75,
            "down_threshold": 25,
        },
    },

    # -------------------------------------------------------
    # 11. OHP (OVERHEAD PRESS)
    # -------------------------------------------------------
    # UWAGA: użytkownik nie wymienił "setup" w swojej liście, ale klasa
    # setup jest niezbędna w przepływie. Dodano klasę 0 = setup automatycznie.
    "ohp": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "arching_back",
            3: "asymmetric",
            4: "flared_elbows",
            5: "half_rep_top",
            6: "half_rep_bottom",
        },
        "num_classes": 7,
        "key_landmarks": {
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "elbow_angle": {
                "joints": ("shoulder", "elbow", "wrist"),
                "correct_range_bottom": (60, 100),
                "correct_range_top": (160, 180),
                "error_name": "niepełny zakres ruchu (łokieć)",
            },
            "shoulder_angle": {
                "joints": ("hip", "shoulder", "elbow"),
                "correct_range_bottom": (40, 70),
                "error_name": "flared elbows (łokcie zbyt szeroko)",
            },
            "back_alignment": {
                "joints": ("shoulder", "mid_hip", "knee"),
                "correct_range": (165, 180),
                "error_name": "wygięte plecy (arching_back)",
            },
        },
        "rep_phases": {
            "angle_joint": ("shoulder", "elbow", "wrist"),
            "up_threshold": 160,
            "down_threshold": 95,
        },
    },

    # -------------------------------------------------------
    # 12. WIOSŁOWANIE W OPADZIE (BENT OVER ROW)
    # -------------------------------------------------------
    "bent_over_row": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "asymmetric",
            3: "flared_elbows",
            4: "half_rep_bottom",
            5: "half_rep_top",
            6: "swinging",
        },
        "num_classes": 7,
        "key_landmarks": {
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "elbow_angle": {
                "joints": ("shoulder", "elbow", "wrist"),
                "correct_range_top": (60, 100),
                "correct_range_bottom": (150, 180),
                "error_name": "niepełny zakres ruchu (łokieć)",
            },
            "shoulder_angle": {
                "joints": ("hip", "shoulder", "elbow"),
                "correct_range": (15, 60),
                "error_name": "flared elbows (łokcie zbyt szeroko)",
            },
            "body_alignment": {
                "joints": ("shoulder", "hip", "knee"),
                "correct_range": (80, 130),
                "error_name": "kołysanie ciałem (swinging)",
            },
        },
        "rep_phases": {
            "angle_joint": ("shoulder", "elbow", "wrist"),
            "up_threshold": 145,
            "down_threshold": 85,
        },
    },

    # -------------------------------------------------------
    # 13. WYCISKANIE FRANCUSKIE (FRENCH PRESS / SKULL CRUSHER)
    # -------------------------------------------------------
    "french_press": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "arms_in_same_line_as_shoulders",
            3: "asymmetric",
            4: "flared_elbows",
            5: "half_rep_bottom",
            6: "half_rep_top",
        },
        "num_classes": 7,
        "key_landmarks": {
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "elbow_angle": {
                "joints": ("shoulder", "elbow", "wrist"),
                "correct_range_top": (160, 180),
                "correct_range_bottom": (40, 70),
                "error_name": "niepełny zakres ruchu (łokieć)",
            },
            "shoulder_angle": {
                "joints": ("hip", "shoulder", "elbow"),
                "correct_range": (155, 180),
                "error_name": "ramię w lini z barkami (arms_in_same_line_as_shoulders)",
            },
            "elbow_flare": {
                "joints": ("sternum", "shoulder", "elbow"),
                "correct_range": (15, 40),
                "error_name": "flared elbows (łokcie zbyt szeroko)",
            },
        },
        "rep_phases": {
            "angle_joint": ("shoulder", "elbow", "wrist"),
            "up_threshold": 155,
            "down_threshold": 75,
        },
    },

    # -------------------------------------------------------
    # 14. ROZPIĘTKI NA KLATĘ (CHEST FLY)
    # -------------------------------------------------------
    "chest_fly": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "bent_arms",
            3: "half_rep_bottom",
            4: "half_rep_top",
            5: "straight_arms",
        },
        "num_classes": 6,
        "key_landmarks": {
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "shoulder_abduction": {
                "joints": ("hip", "shoulder", "elbow"),
                "correct_range_bottom": (15, 40),
                "correct_range_top": (70, 100),
                "error_name": "niepełny zakres ruchu (rozpiętki)",
            },
            "elbow_angle": {
                "joints": ("shoulder", "elbow", "wrist"),
                "correct_range": (145, 170),
                "error_name": "zbyt zgięte/proste ręce",
            },
        },
        "rep_phases": {
            "angle_joint": ("hip", "shoulder", "elbow"),
            "up_threshold": 75,
            "down_threshold": 30,
        },
    },

    # -------------------------------------------------------
    # 15. MUSCLE UP
    # -------------------------------------------------------
    "muscle_up": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "kipping",
            3: "wide_elbows",
            4: "alternating_arms",
        },
        "num_classes": 5,
        "angle_rules": {},
        "rep_phases": {},
    },

    # -------------------------------------------------------
    # 16. L-SIT
    # -------------------------------------------------------
    "l_sit": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "shoulders_not_retracted",
            3: "legs_low",
            4: "bent_legs",
        },
        "num_classes": 5,
        "exercise_type": "isometric",
        "key_landmarks": {
            "nose": 0, "left_ear": 1, "right_ear": 2,
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "left_ankle": 13,    "right_ankle": 14,
            "left_big_toe": 15,  "right_big_toe": 16,
            "left_heel": 17,     "right_heel": 18,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "hip_angle": {
                "joints": ("shoulder", "hip", "knee"),
                "correct_range": (70, 100),
                "error_name": "nogi poniżej 90° (legs_low)",
            },
            "knee_angle": {
                "joints": ("hip", "knee", "ankle"),
                "correct_range": (160, 180),
                "error_name": "zgięte nogi (bent_legs)",
            },
            "shoulder_retraction": {
                "joints": ("hip", "shoulder", "elbow"),
                "correct_range": (70, 120),
                "error_name": "nie spięte barki (shoulders_not_retracted)",
            },
        },
        "rep_phases": {
            "angle_joint": ("shoulder", "hip", "knee"),
            "up_threshold": 90,
            "down_threshold": 80,
        },
    },

    # -------------------------------------------------------
    # 17. WYKROKI (LUNGE)
    # -------------------------------------------------------
    "lunge": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "half_rep_top",
            3: "inward_knee",
        },
        "num_classes": 4,
        "key_landmarks": {
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "left_ankle": 13,    "right_ankle": 14,
            "left_big_toe": 15,  "right_big_toe": 16,
            "left_heel": 17,     "right_heel": 18,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "front_knee": {
                "joints": ("hip", "knee", "ankle"),
                "correct_range_bottom": (80, 100),
                "correct_range_top": (160, 180),
                "error_name": "niepełny zakres ruchu (kolano)",
            },
        },
        "rep_phases": {
            "angle_joint": ("hip", "knee", "ankle"),
            "up_threshold": 150,
            "down_threshold": 100,
        },
    },

    # -------------------------------------------------------
    # 18. HIP THRUST
    # -------------------------------------------------------
    "hip_thrust": {
        "labels": {
            0: "setup",
            1: "arching_back",
            2: "correct",
            3: "foot_too_close_hips",
            4: "half_rep_bottom",
            5: "half_rep_top",
        },
        "num_classes": 6,
        "key_landmarks": {
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "left_ankle": 13,    "right_ankle": 14,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "hip_extension": {
                "joints": ("shoulder", "hip", "knee"),
                "correct_range_top": (160, 180),
                "correct_range_bottom": (90, 130),
                "error_name": "niepełny zakres ruchu (hip)",
            },
            "knee_angle": {
                "joints": ("hip", "knee", "ankle"),
                "correct_range_top": (80, 110),
                "error_name": "stopy za blisko pośladków (foot_too_close_hips)",
            },
            "back_alignment": {
                "joints": ("sternum", "mid_hip", "knee"),
                "correct_range": (155, 180),
                "error_name": "wygięte plecy (arching_back)",
            },
        },
        "rep_phases": {
            "angle_joint": ("shoulder", "hip", "knee"),
            "up_threshold": 155,
            "down_threshold": 115,
        },
    },

    # -------------------------------------------------------
    # 19. PLANK
    # -------------------------------------------------------
    "plank": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "high_hips",
            3: "low_hips",
            4: "no_retraction",
            5: "too_far_back",
            6: "too_forward",
        },
        "num_classes": 7,
        "exercise_type": "isometric",
        "key_landmarks": {
            "nose": 0, "left_ear": 1, "right_ear": 2,
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "left_ankle": 13,    "right_ankle": 14,
            "left_big_toe": 15,  "right_big_toe": 16,
            "left_heel": 17,     "right_heel": 18,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "body_alignment": {
                "joints": ("shoulder", "hip", "ankle"),
                "correct_range": (160, 180),
                "error_name": "biodra zbyt nisko/wysoko",
            },
        },
        "rep_phases": {
            "angle_joint": ("shoulder", "hip", "ankle"),
            "up_threshold": 170,
            "down_threshold": 160,
        },
    },

    # -------------------------------------------------------
    # 20. HOLLOW BODY
    # -------------------------------------------------------
    "hollow_body": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "hidden_chest",
            3: "no_scapula_engagement",
        },
        "num_classes": 4,
        "exercise_type": "isometric",
        "key_landmarks": {
            "nose": 0, "left_ear": 1, "right_ear": 2,
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "left_ankle": 13,    "right_ankle": 14,
            "left_big_toe": 15,  "right_big_toe": 16,
            "left_heel": 17,     "right_heel": 18,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "body_curve": {
                "joints": ("wrist", "hip", "ankle"),
                "correct_range": (140, 170),
            },
        },
        "rep_phases": {
            "angle_joint": ("wrist", "hip", "ankle"),
            "up_threshold": 160,
            "down_threshold": 140,
        },
    },

    # -------------------------------------------------------
    # 21. V-UP
    # -------------------------------------------------------
    "v_up": {
        "labels": {
            0: "setup",
            1: "correct",
            2: "dropping_legs",
            3: "half_rep",
        },
        "num_classes": 4,
        "key_landmarks": {
            "nose": 0, "left_ear": 1, "right_ear": 2,
            "left_shoulder": 3,  "right_shoulder": 4,
            "left_elbow": 5,     "right_elbow": 6,
            "left_wrist": 7,     "right_wrist": 8,
            "left_hip": 9,       "right_hip": 10,
            "left_knee": 11,     "right_knee": 12,
            "left_ankle": 13,    "right_ankle": 14,
            "left_big_toe": 15,  "right_big_toe": 16,
            "left_heel": 17,     "right_heel": 18,
            "sternum": 19,       "mid_hip": 20,
        },
        "angle_rules": {
            "hip_angle": {
                "joints": ("shoulder", "hip", "knee"),
                "correct_range_bottom": (60, 100),
                "correct_range_top": (160, 180),
                "error_name": "niepełny zakres ruchu (V-up)",
            },
        },
        "rep_phases": {
            "angle_joint": ("shoulder", "hip", "knee"),
            "up_threshold": 150,
            "down_threshold": 95,
        },
    },
}

# Zbiór ćwiczeń izometrycznych (czas zamiast powtórzeń)
ISOMETRIC_EXERCISES = {
    name for name, cfg in EXERCISE_CLASSES.items()
    if cfg.get("exercise_type") == "isometric"
}

# ============================================================
# PARAMETRY IZOMETRYCZNE (sliding window)
# ============================================================
# Okno klasyfikacji dla ćwiczeń izometrycznych (w klatkach)
# 30 klatek = 1s przy 30 FPS — klasyfikacja co 1 sekundę
ISOMETRIC_WINDOW_SIZE = SEQUENCE_LENGTH  # 30 klatek = 1s
# Przesunięcie okna (w klatkach) — co ile klatek nowa klasyfikacja
# 15 klatek = 0.5s → okno przesuwa się co pół sekundy (50% overlap)
ISOMETRIC_WINDOW_STRIDE = 15
# Minimalny confidence żeby uznać klasyfikację za pewną
ISOMETRIC_MIN_CONFIDENCE = 0.4

# Aktywne ćwiczenie (domyślnie)
ACTIVE_EXERCISE = "pushup"
NUM_CLASSES = EXERCISE_CLASSES[ACTIVE_EXERCISE]["num_classes"]
