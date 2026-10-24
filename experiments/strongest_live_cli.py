"""One short synthetic CLI with organizer Qwen; credentials process-local only."""
import os,sys,json,time,zipfile,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from verify_package import fixture,check_output
from automl_runtime import child_environment
OUT=ROOT/'results/strongest-team/live-cli.json';WORK=Path('/private/tmp/alfa-strongest-live-cli')
if OUT.exists() or WORK.exists():raise RuntimeError('Preserve previous live evidence, no automatic repeated paid calls')
WORK.mkdir()
archive=ROOT/'dist/alfa_strongest_candidate_v2.zip'
with zipfile.ZipFile(archive) as z:z.extractall(WORK)
metadata=fixture(WORK,'rmse')
for line in (ROOT/'.env.local').read_text().splitlines():
 if '=' in line and line.split('=',1)[0].strip()=='OPENAI_API_KEY':os.environ['OPENAI_API_KEY']=line.split('=',1)[1].strip().strip('\"\'')
os.environ.update(OPENAI_BASE_URL='https://changellenge.up.railway.app/v1',OPENAI_MODEL='qwen/qwen3.8-27b')
env=child_environment(1,include_api=True);env.update(AUTOML_TIME_LIMIT='40',AUTOML_MAX_EXPERIMENTS='2',AUTOML_THREADS='1',MPLCONFIGDIR=str(WORK/'mpl'))
started=time.monotonic()
with (WORK/'cli.log').open('w') as log:result=subprocess.run([sys.executable,'solution.py','task.md'],cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=65)
assert result.returncode==0
summary,submission=check_output(WORK,metadata)
api=[]
for p in WORK.glob('agent_runs/*/llm-*/response.json'):
 record=json.loads(p.read_text());api.append(record.get('result',record))
r={'status':'passed','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'seconds':time.monotonic()-started,'rows':len(submission),'summary_status':summary['status'],'llm_calls':summary['llm_calls'],'api':api,'final_refit':summary.get('final_refit',{}).get('status'),'workdir':str(WORK),'source':'synthetic only, no public data or secrets in report'}
experiments=json.loads(next(WORK.glob('agent_runs/*/experiments.json')).read_text())
r['experiment_sources']=[x['source'] for x in experiments]
r['llm_experiments_completed']=sum(x['source']=='llm' and x['status']=='completed' for x in experiments)
assert r['llm_calls']==1 and len(api)==1 and api[0]['status']=='ok' and r['llm_experiments_completed']==1
OUT.write_text(json.dumps(r,indent=2));print(json.dumps(r))
