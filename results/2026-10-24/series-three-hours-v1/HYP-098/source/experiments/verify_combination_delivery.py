"""Clean flat-package full refit with frozen train-only choices, no validation fits."""
import hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import pandas as pd
from automl_data import load_task,validate_submission
ROOT=Path(__file__).resolve().parents[1]
def main():
    archive=ROOT/'dist/solution_combined_metrics.zip'
    manifest=json.loads((ROOT/'results/2026-10-23/package-manifest-combinations.json').read_text(encoding='utf-8'))
    assert manifest['sha256']==hashlib.sha256(archive.read_bytes()).hexdigest()
    parent=ROOT/'results/2026-10-23'/('combination-delivery-'+str(time.time_ns()));parent.mkdir()
    extracted=ROOT/'tmp'/('combo-flat-'+str(time.time_ns()));extracted.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:z.extractall(extracted)
    records=[]
    for kind in ['regression','multiclass']:
        expected=manifest['baseline'][kind];bundle=load_task(ROOT/'data'/kind/'task.md')
        env=os.environ.copy();env.update(AUTOML_MAX_EXPERIMENTS='0',AUTOML_PORTFOLIO='0',AUTOML_TIME_LIMIT='1200',PYTHONIOENCODING='utf-8')
        env.pop('PYTHONPATH',None)
        for k in list(env):
            if k.startswith(('OPENAI_','SPEC006_','COMBO_')):env.pop(k,None)
        started=time.monotonic()
        p=subprocess.run([sys.executable,'-X','utf8',str(extracted/'solution.py'),str(bundle.task_path)],cwd=extracted,env=env,
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',timeout=1200)
        (parent/(kind+'.log')).write_text(p.stdout,encoding='utf-8');assert p.returncode==0,p.stdout
        path=sorted((bundle.task_path.parent/'agent_runs').glob('*/summary.json'))[-1];summary=json.loads(path.read_text(encoding='utf-8'))
        assert summary['cached_history'] and summary['selected']['strategy']==expected['candidate']['strategy'],summary
        assert abs(summary['selected_score']-expected['score'])<1e-12
        assert not list(path.parent.glob('experiment-*/metrics.json')),'Repeated public validation fits'
        refits=json.loads((path.parent/'initial_refit.json').read_text(encoding='utf-8'))
        assert len(refits)==1 and refits[0]['result']['status']=='completed',refits
        frame=pd.read_csv(bundle.task_path.parent/'submission.csv');validate_submission(bundle,frame)
        audit=json.loads((path.parent/'refit-01/preprocessing-audit.json').read_text(encoding='utf-8'))
        if kind=='multiclass' and 'class_multipliers' in expected['candidate']:
            assert audit.get('class_multipliers',audit.get('calibration_multipliers'))==expected['candidate']['class_multipliers']
            assert audit.get('state_source',audit.get('calibration_source'))=='frozen_candidate' and not audit.get('folds',[])
        if 'blend_weights' in expected['candidate']:assert audit['selection']['selection_source']=='frozen_candidate'
        if 'meta_state' in expected['candidate']:
            assert audit['meta_state']==expected['candidate']['meta_state']
            assert audit['state_source']=='frozen_candidate' and not audit['folds']
        result={'task':kind,'rows':len(frame),'columns':list(frame),'selected_score':summary['selected_score'],
         'strategy':summary['selected']['strategy'],'validation_fit_count':0,'full_refit_count':1,'seconds':time.monotonic()-started,'summary':summary,'audit':audit,
         'submission_sha256':hashlib.sha256((bundle.task_path.parent/'submission.csv').read_bytes()).hexdigest()}
        (parent/(kind+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');records.append(result)
        print(json.dumps({k:result[k] for k in ['task','rows','selected_score','strategy','seconds']},ensure_ascii=False),flush=True)
    result={'status':'passed','archive':str(archive),'archive_sha256':manifest['sha256'],'clean_extraction':str(extracted),
      'records':records,'validation_fit_count':0,'platform_api_tested':False,'hidden_score_verified':False}
    (parent/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'verification':str(parent/'verification.json'),'status':'passed'},ensure_ascii=False))
if __name__=='__main__':main()
