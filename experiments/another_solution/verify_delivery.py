"""Exact platform root-copy execution, no API; smoke budget is not a quality comparison."""
import os,sys,json,time,hashlib,zipfile,shutil,subprocess,signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd,psutil
R=Path(__file__).resolve().parents[2];S=R/'results/another_solution/2026-10-24';manifest=json.loads((R/'reports/PACKAGE_MANIFEST.json').read_text());archive=R/'dist'/manifest['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['sha256']
def run(kind):
 out=S/('clean-zip-'+kind);out.mkdir(exist_ok=False);task=out/'task';task.mkdir()
 with zipfile.ZipFile(archive) as z:
  for name in z.namelist():
   assert Path(name).name==name and (name.endswith('.py') or name=='requirements.txt');(task/name).write_bytes(z.read(name))
 for name in ['train.csv','test.csv','sample_submission.csv','data_dictionary.csv','task.md']:shutil.copyfile(R/'data'/kind/name,task/name)
 env=os.environ.copy()
 for key in list(env):
  if any(w in key.upper() for w in ['API_KEY','BASE_URL','MODEL_NAME','OPENAI_']) or key.startswith(('AUTOML_','COMBO_')):env.pop(key,None)
 env.update(LLM_DISABLED='1',TIME_LIMIT='120',MAX_ITERATIONS='0',TUNE_TRIALS='0',N_JOBS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 started=time.monotonic();peak=0
 print('START exact clean ZIP CLI',kind,flush=True)
 with (out/'command.log').open('w') as log:
  p=subprocess.Popen([sys.executable,'solution.py','task.md'],cwd=task,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while p.poll() is None:
   try:
    root=psutil.Process(p.pid);peak=max(peak,sum(a.memory_info().rss for a in [root]+root.children(recursive=True) if a.is_running()))
   except psutil.Error:pass
   if time.monotonic()-started>240:
    os.killpg(p.pid,signal.SIGKILL);raise TimeoutError(kind)
   time.sleep(.2)
 assert p.returncode==0
 log=(out/'command.log').read_text();assert 'агент упал' not in log
 sample=pd.read_csv(task/'sample_submission.csv');pred=pd.read_csv(task/'submission.csv');assert pred.shape==sample.shape and list(pred)==list(sample)
 assert pred.iloc[:,0].tolist()==sample.iloc[:,0].tolist() and pred.notna().all().all()
 if kind=='regression':assert np.isfinite(pred.iloc[:,1].to_numpy(float)).all()
 else:assert set(pred.iloc[:,1]).issubset(set(pd.read_csv(task/'train.csv').attack_cat))
 work=task/'.agent_work';assert (work/'report.md').exists();events=[json.loads(line) for line in (work/'experiments.jsonl').read_text().splitlines()];assert any(v['ok'] for v in events)
 assert peak<32*1024**3
 for n,h in manifest['source_sha256'].items():assert hashlib.sha256((task/n).read_bytes()).hexdigest()==h
 shutil.copyfile(work/'report.md',out/'agent-report.md');shutil.copyfile(work/'experiments.jsonl',out/'agent-experiments.jsonl')
 result={'status':'passed','task':kind,'command':['python3','solution.py','task.md'],'archive_sha256':manifest['sha256'],'root_files_copied':len(manifest['root_files']),'source_matches_manifest':True,'submission_rows':len(pred),'submission_columns':list(pred),'sample_id_order_exact':True,'seconds':time.monotonic()-started,'peak_process_tree_RSS_bytes':peak,'budget_seconds':120,'MAX_ITERATIONS':0,'N_JOBS':4,'llm_disabled':True,'real_LLM_tested':False,'test_targets_available':False,'note':'Technical clean-entrypoint/CSV check; reduced smoke budget, not the 300s quality comparison or a 3h Docker test','submission_sha256':hashlib.sha256((task/'submission.csv').read_bytes()).hexdigest()}
 (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result),flush=True);return result
with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,['regression','multiclass']))
(S/'delivery-verification.json').write_text(json.dumps({'status':'passed','archive_sha256':manifest['sha256'],'cases':results,'platform_submission_sent':False},ensure_ascii=False,indent=2))
