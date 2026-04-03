import os
import shutil
import json
import subprocess

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'kaggle' in os.path.dirname(os.path.abspath(__file__)) else os.path.dirname(os.path.abspath(__file__))

tmp_dir = os.path.join(os.environ.get('TEMP', 'C:\\tmp'), 'kaggle_payload')
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir)

for f in os.listdir(root_dir):
    if f.endswith('.py') or f in ['README.md', 'requirements.txt']:
        src = os.path.join(root_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, tmp_dir)

kaggle_dir = os.path.join(root_dir, 'kaggle')
if os.path.exists(kaggle_dir):
    shutil.copytree(kaggle_dir, os.path.join(tmp_dir, 'kaggle'), dirs_exist_ok=True)

metadata = {
  "title": "cpu1-source-code",
  "id": "ignacioandrsschwindt/cpu1-source-code",
  "licenses": [{"name": "CC0-1.0"}]
}
with open(os.path.join(tmp_dir, 'dataset-metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("Starting Kaggle dataset upload...")
kaggle_exe = os.path.join(root_dir, '.venv', 'Scripts', 'kaggle.exe')
subprocess.run([kaggle_exe, 'datasets', 'version', '-p', tmp_dir, '-m', 'Update source code', '--dir-mode', 'zip'])
