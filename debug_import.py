import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
POSE_MODEL_DIR = r"c:\Users\Adrian\Desktop\Przedsiewziecie_inzynierskie_modele_ai\pose_training"

import importlib.util

def _load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    _pose_config = _load_module_from_file("pose_config", os.path.join(POSE_MODEL_DIR, "config.py"))
    _pose_model = _load_module_from_file("pose_model", os.path.join(POSE_MODEL_DIR, "model.py"))
    _pose_dataset = _load_module_from_file("pose_dataset", os.path.join(POSE_MODEL_DIR, "dataset.py"))
except Exception as e:
    import traceback
    traceback.print_exc()
