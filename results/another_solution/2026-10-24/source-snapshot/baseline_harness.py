"""Runs the UNCHANGED archive agent, saving its otherwise memory-only evidence."""
import os,sys,time,json,hashlib,subprocess,zipfile,shutil,traceback
from pathlib import Path
import numpy as np,joblib,psutil
R=Path('/Users/sergey_sahinov/Desktop/alpha_hack/alfa-hack')
S=R/'results/another_solution/2026-10-24';SOURCE=Path('/Users/sergey_sahinov/Downloads/solution-senior-agent-2 (1) (1).zip')
kind=sys.argv[1];out=S/('original-'+kind);out.mkdir(parents=True,exist_ok=False);clean=out/'task';clean.mkdir()
with zipfile.ZipFile(SOURCE) as z:
 for name in z.namelist():(clean/name).write_bytes(z.read(name))
for name in ['train.csv','test.csv','sample_submission.csv','data_dictionary.csv','task.md']:shutil.copyfile(R/'data'/kind/name,clean/name)
for key in list(os.environ):
 if any(word in key.upper() for word in ['API_KEY','BASE_URL','MODEL_NAME','OPENAI_']) or key.startswith(('AUTOML_','COMBO_')):os.environ.pop(key,None)
os.environ.update(LLM_DISABLED='1',TIME_LIMIT='300',MAX_ITERATIONS='3',TUNE_TRIALS='0',N_JOBS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
os.chdir(clean);sys.path.insert(0,str(clean))
import ag_orchestrator as O,ag_finalize as F,ag_research_memory as MEM
started=time.monotonic()
source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in clean.glob('*.py')}
(out/'evidence.json').write_text(json.dumps({'stage':'original archive baseline before changes','archive_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'source_sha256':source_hashes,'data_sha256':{name:hashlib.sha256((clean/name).read_bytes()).hexdigest() for name in ['train.csv','test.csv','task.md']},'budget':{'TIME_LIMIT':300,'MAX_ITERATIONS':3,'N_JOBS':4,'TUNE_TRIALS':0,'LLM_DISABLED':1},'branch':subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip()},indent=2))
load=O.Agent.load;add=MEM.Memory.add;report=O.Agent.report

def load_hook(a):
 load(a)
 np.savez_compressed(out/'splits.npz',A=a.A,B=a.B,**{f'fit_{i}':tr for i,(tr,va) in enumerate(a.folds)},**{f'valid_{i}':va for i,(tr,va) in enumerate(a.folds)})
 (out/'task_spec.json').write_text(json.dumps(a.task.short(),ensure_ascii=False,indent=2,default=str))
 (out/'structure.json').write_text(json.dumps(a.structure,ensure_ascii=False,indent=2,default=str))

def add_hook(m,r):
 add(m,r)
 if r.ok:np.savez_compressed(out/f'exp-{r.exp_id:02d}-predictions.npz',oof=r.oof,test_pred=r.test_pred)

def report_hook(a,choice,single,ens):
 report(a,choice,single,ens)
 joblib.dump({'task':a.task,'y':a.ds.y,'A':a.A,'B':a.B,'folds':a.folds,'single':single,'ensemble':ens,'choice':choice,'runs':a._ensemble_candidates(a.memory.best(),single),'config':a.cfg,'structure':a.structure},out/'replay.joblib',compress=3)
 sol=choice.solution
 np.savez_compressed(out/'final-predictions.npz',y=a.ds.y,oof=sol.oof,test_pred=sol.test_pred,A=a.A,B=a.B)
 result={'status':'completed','task':kind,'metric':a.metric.name,'oof_score':sol.cv,'holdout_B_score':sol.holdout_b,'selected':sol.name,'members':sol.members,'post':sol.post,'experiments':[r.summary() for r in a.memory.runs],'seconds':time.monotonic()-started,'real_llm_tested':False,'archived_source_unchanged':all(hashlib.sha256((clean/n).read_bytes()).hexdigest()==h for n,h in source_hashes.items())}
 (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str))
O.Agent.load=load_hook;MEM.Memory.add=add_hook;O.Agent.report=report_hook
try:
 code=O.run(['task.md']);assert code==0
 if not (out/'metrics.json').exists():raise RuntimeError('Agent did not finalize/report; inspect log and experiments')
 result=json.loads((out/'metrics.json').read_text());print('BASELINE_RESULT',json.dumps({k:v for k,v in result.items() if k!='experiments'}),flush=True)
except BaseException as e:
 (out/'failure.json').write_text(json.dumps({'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'seconds':time.monotonic()-started},indent=2));raise
