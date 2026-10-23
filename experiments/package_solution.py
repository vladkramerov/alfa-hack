"""Produce the platform's flat archive without data, models or dependencies."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    provided=Path.home()/'Downloads/requirements (1).txt'
    shutil.copyfile(provided,ROOT/'requirements.txt')
    names=['solution.py','automl_data.py','automl_models.py','automl_history.py','automl_known.py','automl_llm.py','automl_worker.py','requirements.txt']
    destination=ROOT/'dist/solution_ivan_branch.zip'; destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in names: archive.write(ROOT/name,name)
    with zipfile.ZipFile(destination) as archive:
        assert sorted(archive.namelist())==sorted(names)
        assert all('/' not in name and '\\' not in name for name in archive.namelist())
        assert archive.read('requirements.txt')==provided.read_bytes()
    assert 0<destination.stat().st_size<100_000_000
    record={'file':str(destination.relative_to(ROOT)),'bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'files':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},'requirements_exact':True,'no_data_models_secrets':True}
    (ROOT/'results/2026-10-23/package-manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
