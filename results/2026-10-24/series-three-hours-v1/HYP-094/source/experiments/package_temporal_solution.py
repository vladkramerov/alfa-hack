"""New validated archive; all previous delivery archives remain intact."""
import hashlib,json,zipfile
from pathlib import Path
from experiments.package_improved_solution import NAMES
ROOT=Path(__file__).resolve().parents[1]
def main():
    verification=json.loads((ROOT/'results/2026-10-24/series-alternatives-v1/verification.json').read_text(encoding='utf8'))
    assert verification['status']=='passed'
    provided=Path('C:/Users/User/Downloads/requirements (1).txt').read_bytes();assert (ROOT/'requirements.txt').read_bytes()==provided
    names=NAMES+['automl_combo_memory.py','automl_feature_sets.py','automl_reg_combo.py','automl_class_combo.py','automl_combo_policy.py','automl_temporal_memory.py']
    destination=ROOT/'dist/solution_alternative_v2.zip';baseline=json.loads((ROOT/'baseline.json').read_text(encoding='utf8'))
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(ROOT/n,n)
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None and set(z.namelist())==set(names) and len(z.namelist())==16
        assert all('/' not in n and '\\' not in n for n in z.namelist());assert z.read('requirements.txt')==provided
    assert 0<destination.stat().st_size<100_000_000
    manifest={'file':str(destination),'bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
      'root_files':names,'sources':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},'baseline':baseline,
      'provided_requirements_exact':True,'data_or_models_included':False,'hidden_platform_score_verified':False}
    (ROOT/'results/2026-10-24/package-manifest-alternative-v2.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:manifest[k] for k in ['file','bytes','sha256']},ensure_ascii=False))
if __name__=='__main__':main()
