from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
MANIFEST = DATA / 'manifest.json'
MODELS = ROOT / 'models'
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'docs' / 'figures'
FIRMWARE_RX = ROOT / 'src' / 'rx'


def ensure_dirs() -> None:
    for d in (MODELS, RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
