from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import bdtools.config as cfg

print("Proyecto:", ROOT)
print("Paquete:", cfg.__file__)
print("Datos raw:", cfg.RAW_DIR)
print("Archivo esperado:", cfg.RAW_DATA_FILE)
print("Existe raw:", cfg.RAW_DATA_FILE.exists())
