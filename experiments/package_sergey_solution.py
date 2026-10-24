"""Deterministic, strict root-only platform ZIP; no config, models or data."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['solution.py','automl_data.py','automl_models.py','automl_preprocessing.py',
         'automl_history.py','automl_known.py','automl_llm.py','automl_worker.py',
         'automl_advanced.py','automl_combo_memory.py','automl_feature_sets.py',
         'automl_reg_combo.py','automl_class_combo.py','automl_combo_policy.py',
         'automl_temporal_memory.py','automl_panel_memory.py','requirements.txt']


def main():
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip() == 'sergey_branch'
    from automl_known import KNOWN_TASKS
    assert KNOWN_TASKS == [], 'Embedded pretrained public dataset state is forbidden'
    provided = (ROOT.parent/'data_for_participants/requirements.txt').read_bytes()
    assert (ROOT/'requirements.txt').read_bytes() == provided
    archive = ROOT/'dist/solution_sergey_panel_v1.zip'
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as package:
        for name in NAMES:
            assert Path(name).name == name and '/' not in name and '\\' not in name
            assert name.endswith('.py') or name == 'requirements.txt'
            data = (ROOT/name).read_bytes()
            if name.endswith('.py'): compile(data,name,'exec')
            info = zipfile.ZipInfo(name,date_time=(2026,10,24,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            package.writestr(info,data,compresslevel=9)
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert package.namelist() == NAMES
        assert len(set(package.namelist())) == len(NAMES)
        assert not any(info.is_dir() for info in package.infolist())
        assert package.read('requirements.txt') == provided
        assert all(package.read(name) == (ROOT/name).read_bytes() for name in NAMES)
    assert 0 < archive.stat().st_size <= 100_000_000
    manifest = {'archive':archive.name,'bytes':archive.stat().st_size,
        'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'root_files':NAMES,
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in NAMES},
        'root_only':True,'directories':0,'provided_requirements_exact':True,
        'data_models_configs_secrets_included':False,'embedded_pretrained_calibration_or_public_dataset_state':False,'python':'3.12',
        'entrypoint':'python3 solution.py task.md','output':'submission.csv next to task.md',
        'real_llm_api_tested':False,'real_llm_api_skipped_at_user_request':True,
        'hidden_platform_score_verified':False}
    out=ROOT/'results/2026-10-24/sergey-improvement-v1/package-manifest.json'
    out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:manifest[k] for k in ['archive','bytes','sha256','directories']}))


if __name__=='__main__': main()
