import zipfile
import os

# Path to the zip file
zip_file_path = 'data/mnist-inrs.zip'

# Directory to extract the contents
extract_to = 'data/'

# Ensure the file exists
if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted files to {extract_to}")
else:
    print(f"Zip file not found: {zip_file_path}")