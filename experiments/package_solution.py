"""Reproducible flat Boosters archive; run from any working directory."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REQUIREMENTS_SHA256 = '9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6'
MODULES = ('solution.py', 'automl_engine.py', 'automl_data.py', 'automl_models.py',
           'automl_worker.py', 'automl_llm.py', 'automl_runtime.py', 'automl_panel.py', 'requirements.txt')


def build(output=None, source_root=None):
    source = Path(source_root) if source_root else ROOT
    destination = Path(output) if output else ROOT / 'dist' / 'alfa_autoresearch_submission.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256((source / 'requirements.txt').read_bytes()).hexdigest() != OFFICIAL_REQUIREMENTS_SHA256:
        raise ValueError('requirements.txt differs from the original organizer file')
    hashes = {}
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in MODULES:
            content = (source / name).read_bytes()
            if name.endswith('.py'):
                compile(content, name, 'exec')
            hashes[name] = hashlib.sha256(content).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    if destination.stat().st_size > 100_000_000:
        raise ValueError('Archive exceeds 100 MB')
    with zipfile.ZipFile(destination) as archive:
        if archive.namelist() != list(MODULES) or archive.testzip() is not None:
            raise ValueError('Archive validation failed')
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {'archive': destination.name, 'bytes': destination.stat().st_size,
                'sha256': digest, 'files': hashes, 'deterministic': True}
    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--source-root', help='Optional immutable source snapshot for historical reproduction')
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.source_root), indent=2))
