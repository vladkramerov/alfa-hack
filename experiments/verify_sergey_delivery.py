"""Exact platform-copy simulation, final public fits, no real LLM requests."""
import hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
import psutil
from automl_data import load_task,read_submission,validate_submission

ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'results/2026-10-24/sergey-improvement-v1'


def main():
    manifest=json.loads((SUITE/'package-manifest.json').read_text())
    archive=ROOT/'dist'/manifest['archive']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['sha256']
    out=SUITE/'clean-public-delivery';out.mkdir(exist_ok=False)
    baseline=json.loads((ROOT/'baseline.json').read_text())
    records=[]
    for kind in ['regression','multiclass']:
        folder=ROOT/'tmp'/('sergey-platform-'+kind+'-'+str(time.time_ns()));folder.mkdir(parents=True)
        with zipfile.ZipFile(archive) as package:
            # Exactly the whitelist the platform copies, no checkout/config/data artifacts.
            for name in package.namelist():
                assert Path(name).name==name and (name.endswith('.py') or name=='requirements.txt')
                (folder/name).write_bytes(package.read(name))
        original=ROOT/'data'/kind
        for name in ['train.csv','test.csv','sample_submission.csv','data_dictionary.csv','task.md']:
            shutil.copyfile(original/name,folder/name)
        env=os.environ.copy()
        for key in list(env):
            if key.startswith(('OPENAI_','COMBO_','SPEC006_')) or key=='PYTHONPATH':env.pop(key,None)
        env.update(AUTOML_PORTFOLIO='1',AUTOML_MAX_EXPERIMENTS='1',AUTOML_TIME_LIMIT='1200',
                   OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf8')
        started=time.monotonic();peak=0
        print('START clean platform command: '+kind,flush=True)
        with (out/(kind+'.log')).open('w') as log:
            process=subprocess.Popen([sys.executable,'solution.py','task.md'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    parent=psutil.Process(process.pid)
                    peak=max(peak,sum(p.memory_info().rss for p in [parent]+parent.children(recursive=True) if p.is_running()))
                except psutil.Error:pass
                if time.monotonic()-started>1200:
                    process.kill();raise TimeoutError(kind+' exceeded delivery verification limit')
                time.sleep(.2)
        assert process.returncode==0,(out/(kind+'.log')).read_text()
        bundle=load_task(folder/'task.md');frame=read_submission(bundle,folder/'submission.csv');validate_submission(bundle,frame)
        assert len(frame)==(22449 if kind=='regression' else 41166)
        path=next((folder/'agent_runs').glob('*/summary.json'));summary=json.loads(path.read_text())
        assert not summary['cached_history'], 'Final agent must start without embedded dataset state'
        assert summary['selected']['strategy']==baseline[kind]['candidate']['strategy'],summary['selected']
        experiments=list(path.parent.glob('experiment-*/metrics.json'))
        assert len(experiments)==1, 'Expected one fresh train-only validation run'
        assert json.loads(experiments[0].read_text())['status']=='completed'
        refits=json.loads((path.parent/'initial_refit.json').read_text())
        assert len(refits)==0
        refit_metrics=list(path.parent.glob('refit-*/metrics.json'))
        assert len(refit_metrics)==1 and json.loads(refit_metrics[0].read_text())['status']=='completed'
        audit=json.loads((path.parent/'refit-01/preprocessing-audit.json').read_text())
        if kind=='regression':
            assert audit['strict_past_only'] and audit['fixed_blend_weights']==[.5,.5]
            assert audit['dynamics']['validation_targets_used'] is False
        else:
            assert audit['state_source']=='frozen_candidate' and not audit['folds']
            assert 'meta_state' not in json.loads((ROOT/'baseline.json').read_text())[kind]['candidate']
            learned=summary['experiments'][0]['refit_candidate']['meta_state']
            assert audit['meta_state']==learned
            assert summary['experiments'][0]['learned_state']['meta_state']==learned
        assert peak < 32*1024**3
        record={'task':kind,'status':'passed','strategy':summary['selected']['strategy'],
            'fresh_local_validation_score_not_test_score':summary['selected_score'],
            'rows':len(frame),'columns':list(frame),'full_refits':1,'validation_runs_from_scratch':1,
            'seconds':time.monotonic()-started,'peak_process_tree_rss_bytes':peak,
            'command':['python3','solution.py','task.md'],'cwd_contains_task_and_solution':True,
            'clean_directory':str(folder),'source_sha256':manifest['source_sha256'],
            'submission_sha256':hashlib.sha256((folder/'submission.csv').read_bytes()).hexdigest(),
            'summary':summary,'audit':audit}
        (out/(kind+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2))
        shutil.copyfile(folder/'submission.csv',out/(kind+'-submission.csv'))
        records.append(record)
        print(json.dumps({k:record[k] for k in ['task','status','rows','seconds','peak_process_tree_rss_bytes']}),flush=True)
    result={'status':'passed','archive_sha256':manifest['sha256'],'records':records,
            'exact_platform_root_copy_simulated':True,'real_llm_api_tested':False,'platform_submission_sent':False}
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('PASS exact root-only platform simulation',flush=True)


if __name__=='__main__':main()
