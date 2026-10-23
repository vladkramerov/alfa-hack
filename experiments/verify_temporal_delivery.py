"""Clean extraction public inference, no repeated validation fits."""
import hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
from automl_data import load_task,validate_submission
ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'results/2026-10-24/package-manifest-alternative-v2.json').read_text(encoding='utf8'))
    archive=Path(manifest['file']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['sha256']
    parent=ROOT/'results/2026-10-24'/('alternative-delivery-'+str(time.time_ns()));parent.mkdir()
    extracted=ROOT/'tmp'/('alternative-flat-'+str(time.time_ns()));extracted.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:z.extractall(extracted)
    records=[]
    for kind in ['regression','multiclass']:
        expected=manifest['baseline'][kind];bundle=load_task(ROOT/'data'/kind/'task.md')
        env=os.environ.copy();env.update(AUTOML_MAX_EXPERIMENTS='0',AUTOML_PORTFOLIO='0',AUTOML_TIME_LIMIT='1200',PYTHONIOENCODING='utf8')
        env.pop('PYTHONPATH',None)
        for k in list(env):
            if k.startswith(('OPENAI_','SPEC006_','COMBO_')):env.pop(k,None)
        started=time.monotonic()
        p=subprocess.run([sys.executable,'-X','utf8',str(extracted/'solution.py'),str(bundle.task_path)],cwd=extracted,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',timeout=1200)
        (parent/(kind+'.log')).write_text(p.stdout,encoding='utf8');assert p.returncode==0,p.stdout
        path=sorted((bundle.task_path.parent/'agent_runs').glob('*/summary.json'))[-1];summary=json.loads(path.read_text(encoding='utf8'))
        assert summary['cached_history'] and summary['selected']['strategy']==expected['candidate']['strategy'],summary
        assert abs(summary['selected_score']-expected['score'])<1e-12
        assert not list(path.parent.glob('experiment-*/metrics.json')),'Repeated public validation fit'
        refits=json.loads((path.parent/'initial_refit.json').read_text(encoding='utf8'));assert len(refits)==1 and refits[0]['result']['status']=='completed',refits
        from automl_data import read_submission
        frame=read_submission(bundle,bundle.task_path.parent/'submission.csv');validate_submission(bundle,frame)
        assert len(frame)==(22449 if kind=='regression' else 41166)
        audit=json.loads((path.parent/'refit-01/preprocessing-audit.json').read_text(encoding='utf8'))
        if kind=='regression':assert audit['strict_past_only'] and not audit['id_or_weather_as_key'] and audit['base']['feature_numbers']==[29,30,31]
        else:assert audit['state_source']=='frozen_candidate' and not audit['folds'] and audit['meta_state']==expected['candidate']['meta_state']
        result={'task':kind,'strategy':summary['selected']['strategy'],'selected_score':summary['selected_score'],
          'rows':len(frame),'columns':list(frame),'full_refits':1,'validation_fits':0,'seconds':time.monotonic()-started,
          'summary':summary,'audit':audit,'submission_sha256':hashlib.sha256((bundle.task_path.parent/'submission.csv').read_bytes()).hexdigest()}
        (parent/(kind+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');records.append(result)
        print(json.dumps({k:result[k] for k in ['task','strategy','selected_score','rows','seconds']},ensure_ascii=False),flush=True)
    result={'status':'passed','archive':str(archive),'archive_sha256':manifest['sha256'],'clean_extraction':str(extracted),'records':records,'repeated_validation_fits':0,'hidden_score_verified':False,'real_platform_api_tested':False}
    (parent/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'status':'passed','verification':str(parent/'verification.json')},ensure_ascii=False))
if __name__=='__main__':main()
