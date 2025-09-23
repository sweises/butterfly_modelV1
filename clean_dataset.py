import os

root_dir = "data"  # Pfad zu deinem Dataset

for root, dirs, files in os.walk(root_dir):
    for f in files:
        if f.startswith("._") or f.lower().endswith(".ds_store"):
            path = os.path.join(root, f)
            print("Lösche:", path)
            os.remove(path)
