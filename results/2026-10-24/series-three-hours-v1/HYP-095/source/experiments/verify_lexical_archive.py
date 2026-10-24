"""Real evaluate→refit→CSV from a clean ZIP, on synthetic opaque IDs/classes."""
import hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    archive=ROOT/'dist/solution_alternative_v2.zip';out=ROOT/'results/2026-10-24'/('lexical-archive-'+str(time.time_ns()));out.mkdir()
    extracted=ROOT/'tmp'/('lexical-flat-'+str(time.time_ns()));extracted.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:z.extractall(extracted)
    nested=out/'nested/inputs';nested.mkdir(parents=True)
    n=80;labels=np.array(['001','010']*(n//2));ids=['0007','0012','0100']
    pd.DataFrame({'key':np.arange(n),'x':np.arange(n),'flag':np.arange(n)%2,'answer':labels}).to_csv(nested/'train.csv',index=False)
    pd.DataFrame({'key':ids,'x':[81,82,83],'flag':[1,0,1]}).to_csv(nested/'test.csv',index=False)
    pd.DataFrame({'key':ids,'answer':['001']*3}).to_csv(nested/'sample_submission.csv',index=False)
    pd.DataFrame({'feature':['key','x','flag','answer'],'role':['id','feature','feature','target'],'dtype':['str','numeric','numeric','str']}).to_csv(nested/'data_dictionary.csv',index=False)
    task=out/'task.md';task.write_text('Classification. Primary metric: Macro F1.\n`nested/inputs/train.csv` `nested/inputs/test.csv` `nested/inputs/sample_submission.csv` `nested/inputs/data_dictionary.csv`.\n',encoding='utf8')
    env=os.environ.copy();env.update(AUTOML_MAX_EXPERIMENTS='1',AUTOML_PORTFOLIO='1',AUTOML_TIME_LIMIT='120',PYTHONIOENCODING='utf8')
    env.pop('PYTHONPATH',None)
    for k in list(env):
        if k.startswith(('OPENAI_','COMBO_','SPEC006_')):env.pop(k,None)
    started=time.monotonic();p=subprocess.run([sys.executable,'-X','utf8',str(extracted/'solution.py'),str(task)],cwd=extracted,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',timeout=120)
    (out/'command.log').write_text(p.stdout,encoding='utf8');assert p.returncode==0,p.stdout
    frame=pd.read_csv(out/'submission.csv',dtype=str,keep_default_na=False)
    assert frame.key.tolist()==ids and set(frame.answer).issubset(set(labels))
    summary=json.loads(next((out/'agent_runs').glob('*/summary.json')).read_text(encoding='utf8'))
    assert summary['selected'] and summary['selected_score'] is not None
    assert len(summary['experiments'])==1 and summary['experiments'][0]['status']=='completed'
    result={'status':'passed','archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
       'pipeline':'synthetic_evaluate_refit_atomic_final_read','rows':3,'original_ids':ids,'valid_lexical_classes':['001','010'],
       'output_ids':frame.key.tolist(),'output_labels':frame.answer.tolist(),'seconds':time.monotonic()-started,
       'synthetic_only':True,'public_validation_fits':0,'new_public_metrics':0,'summary':summary}
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'status':'passed','verification':str(out/'verification.json'),'seconds':result['seconds']},ensure_ascii=False))
if __name__=='__main__':main()
