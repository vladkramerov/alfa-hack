"""Package only verified portable source, then test a clean extraction."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'deliveries/sergey_ivan_fusion_v1'
SUITE=ROOT/'results/2026-10-24/series-sergey-ivan-fusion-v1'
REFIT=ROOT/'results/2026-10-24/HYP-107-03-technical-delivery-refit'
ZIP=ROOT/'dist/solution_sergey_ivan_fusion_v1.zip'
CLEAN=ROOT/'tmp/sergey-ivan-fusion-flat-v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    outcome=read(REFIT/'outcome.json')
    assert outcome['status']=='completed' and outcome['submission_validated'] and outcome['quality_metric_computed'] is False
    assert read(SUITE/'independent-verification.json')['status']=='passed'
    manifest=read(SUITE/'delivery-source-manifest.json')
    for name,digest in manifest['files'].items():assert sha(SOURCE/name)==digest,name
    assert manifest['files']['requirements.txt']=='9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6'
    # Classification and transport/IO/API modules remain byte-identical to Sergey.
    unchanged=[]
    with zipfile.ZipFile(ROOT/'dist/solution_sergey_panel_v1_24e3311.zip') as base:
        for name in base.namelist():
            if name not in ['solution.py','automl_models.py','automl_history.py']:
                assert (SOURCE/name).read_bytes()==base.read(name),name
                unchanged.append(name)
    assert not ZIP.exists() and not CLEAN.exists()
    with zipfile.ZipFile(ZIP,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in manifest['files']:archive.write(SOURCE/name,name)
    assert ZIP.stat().st_size<100_000_000
    CLEAN.mkdir()
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert '/' not in name and '\\' not in name and (name.endswith('.py') or name=='requirements.txt')
        archive.extractall(CLEAN)
    modules=[Path(name).stem for name in manifest['files'] if name.endswith('.py')]
    env=os.environ.copy()
    for name in list(env):
        if 'API_KEY' in name:env.pop(name,None)
    code='''import importlib,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
names=json.loads(sys.argv[2])
for name in names:
    module=importlib.import_module(name)
    assert Path(module.__file__).parent==Path(sys.argv[1]),name
from automl_known import KNOWN_TASKS
assert KNOWN_TASKS==[]
from automl_models import canonical_candidate,build_pipeline
from automl_history import compatible_policy
c=canonical_candidate({'strategy':'sergey_ivan_fusion','family':'extra_trees','parameters':{}})
assert compatible_policy(c)
assert type(build_pipeline('regression',c)).__name__=='SergeyIvanFusion'
print(json.dumps({'all_root_modules_imported':len(names),'known_tasks':0,'candidate_dispatch':True}))
'''
    imports=subprocess.run([sys.executable,'-I','-X','utf8','-c',code,str(CLEAN),json.dumps(modules)],
        cwd=CLEAN,env=env,capture_output=True,text=True,encoding='utf8',timeout=60)
    (SUITE/'clean-import.log').write_text(imports.stdout+imports.stderr,encoding='utf8')
    assert imports.returncode==0,imports.stderr
    # Load the technical full-fit model through the clean ZIP modules, without fitting.
    replay='''import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import numpy as np,joblib
from automl_data import load_task
from automl_sergey_fusion import SergeyIvanFusion
def forbidden(*a,**kw):raise AssertionError('No fitting during delivery replay')
SergeyIvanFusion.fit=forbidden
model=joblib.load(Path(sys.argv[2])/'model.joblib')
bundle=load_task(sys.argv[3])
prediction=model.predict(bundle.X_test)
with np.load(Path(sys.argv[2])/'predictions.npz',allow_pickle=False) as z:expected=z['prediction']
np.testing.assert_allclose(prediction,expected,rtol=0,atol=1e-10)
print(json.dumps({'full_test_prediction_replay':True,'test_rows':len(prediction),'model_fits':0}))
'''
    replay_result=subprocess.run([sys.executable,'-I','-X','utf8','-c',replay,str(CLEAN),str(REFIT),
        str(ROOT/'data/regression/task.md')],cwd=CLEAN,env=env,capture_output=True,text=True,encoding='utf8',timeout=120)
    (SUITE/'clean-replay.log').write_text(replay_result.stdout+replay_result.stderr,encoding='utf8')
    assert replay_result.returncode==0,replay_result.stderr
    # Complete CLI smoke on a nested synthetic task; zero search/model fits.
    taskdir=CLEAN/'cli-case';(taskdir/'nested').mkdir(parents=True)
    (taskdir/'nested/train.csv').write_text('object_id,feature,goal\n1,1,4\n2,2,5\n3,3,6\n4,4,7\n5,5,8\n6,6,9\n7,7,10\n8,8,11\n',encoding='utf8')
    (taskdir/'nested/test.csv').write_text('object_id,feature\n20,2\n21,3\n',encoding='utf8')
    (taskdir/'nested/sample_submission.csv').write_text('object_id,goal\n20,0\n21,0\n',encoding='utf8')
    (taskdir/'nested/data_dictionary.csv').write_text('column,description\nobject_id,identifier\nfeature,numeric\ngoal,target\n',encoding='utf8')
    task=taskdir/'task.md';task.write_text('Regression. Target: goal. ID: object_id. Metric: RMSE.\n'
        'train: nested/train.csv\ntest: nested/test.csv\nsample_submission: nested/sample_submission.csv\n'
        'data_dictionary: nested/data_dictionary.csv\nSave submission.csv next to task.md.\n',encoding='utf8')
    env.update(AUTOML_MAX_EXPERIMENTS='0',AUTOML_PORTFOLIO='0',AUTOML_TIME_LIMIT='60')
    cli=subprocess.run([sys.executable,'-X','utf8',str(CLEAN/'solution.py'),str(task)],
        cwd=ROOT/'tmp',env=env,capture_output=True,text=True,encoding='utf8',timeout=60)
    (SUITE/'clean-cli.log').write_text(cli.stdout+cli.stderr,encoding='utf8')
    assert cli.returncode==0,cli.stderr
    assert (taskdir/'submission.csv').is_file()
    result={'status':'passed','zip':str(ZIP.relative_to(ROOT)),'bytes':ZIP.stat().st_size,'sha256':sha(ZIP),
        'root_files':len(manifest['files']),'module_imports':json.loads(imports.stdout),
        'test_prediction_replay':json.loads(replay_result.stdout),
        'classification_unchanged':True,'unchanged_base_files':unchanged,'requirements_exact':True,
        'clean_nested_cli':True,'runtime_real_api_tested':False,'platform_score_verified':False,
        'technical_refit_seconds':outcome['seconds'],'embedded_learned_state':False}
    target=SUITE/'delivery-verification.json';assert not target.exists()
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
