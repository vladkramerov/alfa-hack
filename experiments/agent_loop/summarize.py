"""Compact process/prediction evidence; no training or API calls."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
variants=['control-live60','candidate-al123','candidate-al123456']
rows=[]
for variant in variants:
 p=ROOT/'results/agent-loop'/(variant+'.json')
 if not p.exists():continue
 report=json.loads(p.read_text())
 for r in report['runs']:
  a=r.get('agent') or {};requests=r.get('llm_requests',[]);exps=a.get('experiments',[])
  decisions={x['exp_id']:x['accepted'] for x in r.get('decisions',[])}
  proposed=[e for e in exps if e['source']=='llm']
  row={'variant':variant,'task':r['task'],'external_score':r.get('score'),'seconds':r['seconds'],'peak_rss_mb':r['peak_rss_mb'],
       'submission_valid':r.get('submission_valid'),'source_unchanged':r['source_unchanged'],'split_key':r['split_key'],
       'zip_sha256':report['zip_sha256'],'submission_sha256':r.get('submission_sha256'),
       'experiment_runs':len(exps),'successful_experiments':sum(e['ok'] for e in exps),
       'llm_proposals_executed':len(proposed),'llm_successful_experiments':sum(e['ok'] for e in proposed),
       'llm_incumbent_improvements':sum(decisions.get(e['exp_id'],False) for e in proposed),
       'requests':len(requests),'planning_requests':sum(e['phase']=='plan' for e in requests),
       'tokens':sum(e.get('tokens_in',0)+e.get('tokens_out',0) for e in requests),
       'llm_wait_seconds':sum(e.get('seconds',0) for e in requests),
       'json_valid':sum(e['valid'] for e in r.get('json_responses',[])),
       'json_attempts':len(r.get('json_responses',[])),
       'internal_oof_score':a.get('oof_score'),'internal_B':a.get('holdout_B'),
       'failed_experiments':[{'exp_id':e['exp_id'],'source':e['source'],'error':e['error']} for e in exps if not e['ok']],
       'skipped':a.get('skipped',[])}
  rows.append(row)
output={'protocol':'600s/8CPU/60iterations/20k tokens/real Qwen; repeated research outer, not independent confirmation',
        'rows':rows,'native_fit_count':'Not recorded; experiment runs may contain multiple CV/refit/tuning fits.'}
(ROOT/'results/agent-loop/comparison.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
for r in rows:print(r['variant'],r['task'],r['external_score'],round(r['seconds'],1),r['tokens'],r['llm_incumbent_improvements'])
