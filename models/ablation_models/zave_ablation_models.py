r"""
Genera y guarda los pesos iniciales (sin entrenar) de todos los modelos de ablation.
Util para benchmarks de velocidad/memoria sin necesidad de entrenamiento previo.

Uso (desde cualquier directorio):
    .venv\Scripts\python.exe models/ablation_models/zave_ablation_models.py
"""
import sys
import json
import importlib.util
from pathlib import Path

import torch

# El script vive en  <root>/models/ablation_models/
# Por eso necesitamos subir dos niveles para llegar a la raiz del proyecto
THIS_FILE   = Path(__file__).resolve()
ABLATION_DIR = THIS_FILE.parent          # .../models/ablation_models/
ROOT_DIR     = THIS_FILE.parent.parent.parent  # .../1.58bits/

# Agregar raiz al path para que los imports de ablation.py / model.py funcionen
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def save_model(py_file: Path) -> None:
    """Instancia y guarda un modelo de ablation en su propia subcarpeta."""
    model_name = py_file.stem
    out_dir    = ABLATION_DIR / model_name
    out_dir.mkdir(exist_ok=True)

    ckpt_path   = out_dir / "checkpoint.pt"
    config_path = out_dir / "config.json"

    if ckpt_path.exists():
        print(f"  [skip] {model_name} — checkpoint.pt ya existe.")
        return

    print(f"  [build] {model_name}...")

    # Importar el modulo run_XX.py dinamicamente
    spec   = importlib.util.spec_from_file_location(model_name, py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # build() construye el modelo con pesos aleatorios (sin entrenar)
    model = module.build()

    # Guardar pesos
    torch.save(model.state_dict(), ckpt_path)
    print(f"    -> checkpoint.pt ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

    # Guardar config serializable (excluir valores no-serializables)
    if hasattr(module, "CONFIG"):
        def _serializable(v):
            return isinstance(v, (int, float, str, bool, list, dict, type(None)))
        cfg = {k: v for k, v in module.CONFIG.items() if _serializable(v)}
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"    -> config.json")


def main():
    SKIP = {"__init__.py", THIS_FILE.name}  # no procesar este mismo archivo

    py_files = sorted(
        f for f in ABLATION_DIR.glob("*.py")
        if f.name not in SKIP
    )

    print(f"Encontrados {len(py_files)} modelos en {ABLATION_DIR}\n")
    errors = []
    for py_file in py_files:
        try:
            save_model(py_file)
        except Exception as e:
            print(f"  [ERROR] {py_file.name}: {e}")
            errors.append((py_file.name, str(e)))

    print(f"\nListo. {len(py_files) - len(errors)}/{len(py_files)} modelos guardados.")
    if errors:
        print("Errores:")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
