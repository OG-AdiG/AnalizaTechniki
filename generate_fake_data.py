import numpy as np
import os

def generate_sample(path):
    # (30 klatek, 21 punktów, 3 wartości: x,y,confidence)
    data = np.random.rand(30, 21, 3)
    np.save(path, data)

# Foldery
base_path = "data/keypoints/pushup"
classes = ["correct", "sagging_hips"]

for cls in classes:
    folder = os.path.join(base_path, cls)
    os.makedirs(folder, exist_ok=True)
    
    # 2 sample na klasę
    for i in range(5):
        generate_sample(os.path.join(folder, f"sample_{i}.npy"))

print("✔ Fake dane wygenerowane")