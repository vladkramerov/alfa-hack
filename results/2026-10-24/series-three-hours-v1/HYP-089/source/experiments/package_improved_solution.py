"""Flat measured-winner delivery, with exact provided requirements and no data."""
import hashlib,json,zipfile
from pathlib import Path
from experiments.run_twenty_hypotheses import ROOT

NAMES=['solution.py','automl_data.py','automl_models.py','automl_preprocessing.py','automl_history.py',
       'automl_known.py','automl_llm.py','automl_worker.py','automl_advanced.py','requirements.txt']

def main():
    provided=Path('C:/Users/User/Downloads/requirements (1).txt')
    if provided.read_bytes()!=(ROOT/'requirements.txt').read_bytes(): raise ValueError('Provided requirements changed')
    destination=ROOT/'dist/solution_improved_metrics.zip'; destination.parent.mkdir(exist_ok=True)
    verification=json.loads((ROOT/'results/2026-10-23/series-twenty-hypotheses-v1/independent-verification.json').read_text(encoding='utf-8'))
    if verification['status']!='passed' or verification['pending_cases']: raise ValueError('Unverified experiment evidence')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as package:
        for name in NAMES: package.write(ROOT/name,name)
    with zipfile.ZipFile(destination) as package:
        assert sorted(package.namelist())==sorted(NAMES)
        assert all('/' not in name and '\\' not in name for name in package.namelist())
        assert package.read('requirements.txt')==provided.read_bytes()
        assert package.testzip() is None
    assert 0<destination.stat().st_size<100_000_000
    result={'file':str(destination),'bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'root_files':NAMES,'sources':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in NAMES},
            'provided_requirements_exact':True,'data_or_models_included':False,'metric_source':'results/2026-10-23/TWENTY_HYPOTHESES_REPORT.md',
            'cached_metrics':{'regression_rmse':41.480963027907194,'classification_macro_f1':.6286047899141216},
            'platform_score_verified':False}
    path=ROOT/'results/2026-10-23/package-manifest-improved.json'
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['file','bytes','sha256','cached_metrics']},ensure_ascii=False))

if __name__=='__main__': main()
