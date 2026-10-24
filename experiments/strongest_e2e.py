"""Frozen strongest-team CLI comparison, preserving completed arms for resume."""
import sys,os,json,time,hashlib,zipfile,subprocess,signal,shutil
from pathlib import Path
import psutil,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from automl_data import load_task,validation_split,score,split_key
from automl_runtime import child_environment
revision=sys.argv[1] if len(sys.argv)>1 else ''
assert revision in ['', '-v2']
WORK=Path('/private/tmp/alfa-strongest-e2e'+revision);WORK.mkdir(exist_ok=True)
OUT=ROOT/('results/strongest-team/e2e'+revision+'.json');ZIP=ROOT/('dist/alfa_strongest_candidate'+revision.replace('-','_')+'.zip');digest=hashlib.sha256(ZIP.read_bytes()).hexdigest()
if OUT.exists():
 report=json.loads(OUT.read_text());assert report['zip_sha256']==digest
else:
 report={'status':'running','zip_sha256':digest,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'runs':[],
         'control_report':'results/score-recovery/e2e-final.json','control_zip_sha256':'e12d691382bc280fae5b2063e2a28ca2d21a18847f1c23f786522dafd464a641',
         'protocol':'Fresh frozen CLI candidate; previous completed controls reused. Equal8CPU/1200s/cap16 shadow;1800s/default original. No API for controlled ML comparison; external outer repeatedly viewed.'}
def save():OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False))
for task,folder in [('regression','Регрессия'),('classification','Многоклассовая классификация')]:
 b=load_task(ROOT/'drive-download-20261023T201455Z-1-001'/folder/'task.md');fit,valid,_=validation_split(b)
 for scope in ['shadow','original']:
  if any(r['task']==task and r['scope']==scope for r in report['runs']):continue
  if task=='regression' and scope=='original' and report['runs'][0]['score']>=37.57832956234321:
   report['regression_rejected']=True;save();break
  directory=WORK/f'{task}-{scope}';directory.mkdir(exist_ok=True)
  with zipfile.ZipFile(ZIP) as z:z.extractall(directory)
  if scope=='shadow':
   if not (directory/'task.md').exists():
    b.train.iloc[fit].to_csv(directory/'train.csv',index=False)
    b.train.iloc[valid].drop(columns=[b.target]).to_csv(directory/'test.csv',index=False)
    sample=b.train.iloc[valid][[b.id_column]].copy() if b.id_column else pd.DataFrame(index=range(len(valid)))
    sample[b.target]=b.y.iloc[fit].mode().iloc[0];sample.to_csv(directory/'sample_submission.csv',index=False)
    (directory/'data_dictionary.csv').write_text('column,description\n')
    (directory/'task.md').write_text(f'Primary metric: {b.metric}\nTarget: {b.target}\nIdentifier: {b.id_column}\nTraining data: train.csv\nTest data: test.csv\nSample submission: sample_submission.csv\nData dictionary: data_dictionary.csv\n')
  else:
   for name in ['train.csv','test.csv','sample_submission.csv','data_dictionary.csv','task.md']:
    if not (directory/name).exists():shutil.copy2(b.task_path.parent/name,directory/name)
  budget=1200 if scope=='shadow' else 1800
  env=child_environment(8);env.update(AUTOML_TIME_LIMIT=str(budget),AUTOML_THREADS='8',MPLCONFIGDIR=str(WORK/'matplotlib'))
  if scope=='shadow':env['AUTOML_MAX_EXPERIMENTS']='16'
  save();started=time.monotonic();peak=0
  summaries=list(directory.glob('agent_runs/*/summary.json'))
  if not summaries:
   if (directory/'process-state.json').exists():
    state=json.loads((directory/'process-state.json').read_text());process=psutil.Process(state['pid']);assert Path(process.cwd())==directory
    child=None
   else:
    with (directory/'cli.log').open('w') as log:child=subprocess.Popen([sys.executable,'solution.py','task.md'],cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    process=psutil.Process(child.pid);(directory/'process-state.json').write_text(json.dumps({'pid':child.pid,'created':time.time()}))
   while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
    try:peak=max(peak,sum(p.memory_info().rss for p in [process]+process.children(recursive=True) if p.is_running())/1024**2)
    except psutil.Error:pass
    if time.monotonic()-started>budget+40 or peak>32000:os.killpg(process.pid,signal.SIGKILL);raise RuntimeError('Exceeded platform budget')
    time.sleep(.5)
   if child is not None:assert child.wait()==0
   summaries=list(directory.glob('agent_runs/*/summary.json'))
  summary=json.loads(summaries[-1].read_text());sub=pd.read_csv(directory/'submission.csv');sample=pd.read_csv(directory/'sample_submission.csv')
  assert list(sub)==list(sample) and len(sub)==len(sample) and not sub.isna().any().any()
  if b.id_column:assert sub[b.id_column].equals(sample[b.id_column])
  value=score(b.metric,b.y.iloc[valid],sub[b.target].to_numpy(),classes=b.classes) if scope=='shadow' else None
  r={'task':task,'scope':scope,'score':value,'key':split_key(fit,valid) if scope=='shadow' else None,
     'seconds':time.monotonic()-started,'peak_rss_mb':peak,'rows':len(sub),'selected':summary['selected'],'summary':str(summaries[-1]),
     'submission_sha256':hashlib.sha256((directory/'submission.csv').read_bytes()).hexdigest(),'zip_sha256':digest}
  report['runs'].append(r);save();print(json.dumps(r,ensure_ascii=False),flush=True)
report['status']='completed';save()
