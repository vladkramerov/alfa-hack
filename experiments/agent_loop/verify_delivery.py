"""Exact ZIP CLI smoke: original public tasks + unknown synthetic regression/classification.

No credentials/API. This checks deployment/fallback, not a quality benchmark.
Run only after sequential live comparisons have stopped.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time,zipfile
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
PRIMARY=Path('/Users/maxchibisov/Projects/HACKS/alfa-hack')
def main():
 p=argparse.ArgumentParser();p.add_argument('zip');p.add_argument('--name',default='delivery');a=p.parse_args();archive=Path(a.zip).resolve()
 work=Path('/private/tmp/alfa-another-live')/a.name;work.mkdir(exist_ok=False)
 results=[];rng=np.random.default_rng(1984)
 for kind,folder in [('regression','Регрессия'),('classification','Многоклассовая классификация'),('synthetic-regression',None),('synthetic-classification',None)]:
  d=work/kind;d.mkdir()
  with zipfile.ZipFile(archive) as z:
   assert z.testzip() is None
   for n in z.namelist():
    assert '/' not in n and (n.endswith('.py') or n=='requirements.txt');(d/n).write_bytes(z.read(n))
  if folder:
   src=PRIMARY/'drive-download-20261023T201455Z-1-001'/folder
   for n in ['train.csv','test.csv','sample_submission.csv','data_dictionary.csv','task.md']:shutil.copy2(src/n,d/n)
  else:
   n=1200;nt=100;X=rng.normal(size=(n+nt,4));target=X[:n,0]*2-X[:n,1]+rng.normal(size=n)*.2
   if kind.endswith('classification'):target=np.where(target>1,'zebra',np.where(target< -1,'amber','violet'))
   frame=pd.DataFrame(X,columns=['m0','m1','m2','m3']);frame['category']=np.where(X[:,3]>0,'north','south');frame.insert(0,'uid',np.arange(n+nt))
   tr=frame.iloc[:n].copy();tr['response']=target;tr.to_csv(d/'train.csv',index=False);frame.iloc[n:].to_csv(d/'test.csv',index=False)
   pd.DataFrame({'uid':frame.uid.iloc[n:],'response':target[0]}).to_csv(d/'sample_submission.csv',index=False)
   metric='macro_f1' if kind.endswith('classification') else 'rmse'
   (d/'task.md').write_text(f'Primary metric: {metric}\nTarget: response\nIdentifier: uid\nTraining data: train.csv\nTest data: test.csv\nSample submission: sample_submission.csv\n')
  env={k:v for k,v in os.environ.items() if not k.startswith(('OPENAI_','LLM_','AUTOML_','COMBO_'))}
  env.update(LLM_DISABLED='1',TIME_LIMIT='120',MAX_ITERATIONS='0',TUNE_TRIALS='0',N_JOBS='8',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8')
  started=time.monotonic()
  with (d/'command.log').open('w') as log:
   r=subprocess.run([os.sys.executable,'solution.py','task.md'],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=150)
  assert r.returncode==0,(kind,r.returncode)
  sub=pd.read_csv(d/'submission.csv');sample=pd.read_csv(d/'sample_submission.csv');train=pd.read_csv(d/'train.csv')
  assert sub.shape==sample.shape and list(sub)==list(sample) and sub.iloc[:,0].equals(sample.iloc[:,0]) and sub.notna().all().all()
  if kind.endswith('classification'):assert set(sub.iloc[:,1])<=set(train[sub.columns[1]])
  else:assert np.isfinite(sub.iloc[:,1].to_numpy(float)).all()
  assert (d/'.agent_work/report.md').exists()
  result={'task':kind,'status':'passed','seconds':time.monotonic()-started,'rows':len(sub),'columns':list(sub),'id_order_exact':True,
          'command':['python3','solution.py','task.md'],'llm_disabled':True,'source_matches_zip':True,
          'submission_sha256':hashlib.sha256((d/'submission.csv').read_bytes()).hexdigest()}
  with zipfile.ZipFile(archive) as z:assert all((d/n).read_bytes()==z.read(n) for n in z.namelist())
  if not folder:
   truth=X[n:,0]*2-X[n:,1]
   if kind.endswith('classification'):
    from sklearn.metrics import f1_score
    truth=np.where(truth>1,'zebra',np.where(truth< -1,'amber','violet'))
    value=float(f1_score(truth,sub.iloc[:,1],average='macro'));assert value>.8
    result['synthetic_macro_f1']=value
   else:
    value=float(np.sqrt(np.mean((truth-sub.iloc[:,1].to_numpy())**2)));assert value<1.
    result['synthetic_rmse']=value
  results.append(result);print(json.dumps(result),flush=True)
 output={'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'checks':results,'platform_uploaded':False,'note':'Reduced-budget CLI technical checks; real-Qwen research measured separately; not 3h Linux container test.'}
 (ROOT/'results/agent-loop'/f'{a.name}.json').write_text(json.dumps(output,indent=2));
if __name__=='__main__':main()
