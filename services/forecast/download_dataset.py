import os
import shutil
import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")

print("Path to dataset files:", path)

# Copy CSV files to local directory
dest_dir = os.path.dirname(os.path.abspath(__file__))
for file_name in os.listdir(path):
    if file_name.endswith(".csv"):
        src_file = os.path.join(path, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        shutil.copy2(src_file, dest_file)
        print(f"Saved local CSV file to: {dest_file}")
