"""Independent saved-output/inference verification; no training."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from automl_data import load_task,split_key
from paper_neural import PLEPreprocessor,PaperNeural,probabilities,tensors

ROOT=Path('/Users/sergey_sahinov/Desktop/alpha_hack/alfa-hack')
assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='sergey_branch'
torch.set_num_threads(4)
b=load_task(ROOT/'data/multiclass/task.md')
groups=pd.util.hash_pandas_object(b.X,index=False).to_numpy()
splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(b.X,b.y,groups))
all_results=[]
for number in [79,80]:
 for stage_index,stage in enumerate(['primary','secondary']):
  out=ROOT/f'results/2026-10-24/HYP-{number:03d}-{stage_index+1:02d}-neural-{stage}'
  if not (out/'metrics.json').exists():continue
  m=json.loads((out/'metrics.json').read_text())
  if m['status']!='completed':continue
  evidence=json.loads((out/'evidence.json').read_text())
  saved=np.load(out/'validation_predictions.npz',allow_pickle=False)
  s=np.load(out/'splits.npz',allow_pickle=False)
  checks=[]
  def check(name,truth):
   checks.append(dict(name=name,passed=bool(truth)))
   if not truth:raise AssertionError(name)
  check('official_train_sha',b.hashes['train.csv']=='171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0')
  check('all_data_sha',b.hashes==evidence['hashes'])
  check('saved_ledger_sha',hashlib.sha256((out/'ledger_before_fit.md').read_bytes()).hexdigest()==evidence['ledger_sha256_before_fit'])
  for name,sha in evidence['source_sha256'].items():
   check('source_'+name,hashlib.sha256((out/name).read_bytes()).hexdigest()==sha)
  fit,valid=splits[stage_index]
  check('exact_fit_split',np.array_equal(s['fit'],fit))
  check('exact_outer_split',np.array_equal(s['valid'],valid))
  check('prediction_positions',np.array_equal(saved['positions'],valid))
  check('split_sha',split_key(fit,valid)==evidence['split_key'])
  check('inner_split_sha',split_key(s['inner_fit'],s['inner_holdout'])==evidence['inner_split_key'])
  check('outer_groups_disjoint',not set(groups[fit])&set(groups[valid]))
  check('inner_groups_disjoint',not set(groups[s['inner_fit']])&set(groups[s['inner_holdout']]))
  check('inner_partition_exact',set(s['inner_fit'])|set(s['inner_holdout'])==set(fit))
  check('all_predictions_string_safe',saved['y'].dtype.kind=='U' and saved['prediction'].dtype.kind=='U' and saved['classes'].dtype.kind=='U')
  check('outer_labels_exact',np.array_equal(saved['y'],b.y.iloc[valid].to_numpy(dtype=str)))
  check('classes_from_fit_only',np.array_equal(saved['classes'],np.unique(b.y.iloc[fit].to_numpy(dtype=str))))
  prob=saved['probabilities']
  check('probability_shape',prob.shape==(len(valid),len(saved['classes'])))
  check('finite_nonnegative_probabilities',np.isfinite(prob).all() and (prob>=0).all())
  check('probability_normalization',np.allclose(prob.sum(1),1,rtol=1e-6,atol=1e-6))
  check('argmax_prediction',np.array_equal(saved['prediction'],saved['classes'][prob.argmax(1)]))
  check('independent_macro_f1',abs(float(f1_score(saved['y'],saved['prediction'],average='macro',zero_division=0))-m['score'])<1e-14)
  check('independent_accuracy',abs(float(accuracy_score(saved['y'],saved['prediction']))-m['accuracy'])<1e-14)
  check('independent_weighted_f1',abs(float(f1_score(saved['y'],saved['prediction'],average='weighted',zero_division=0))-m['weighted_f1'])<1e-14)
  trace=[json.loads(line) for line in (out/'epoch_trace.jsonl').read_text().splitlines()]
  inner=[r for r in trace if r['phase']=='inner']
  refit=[r for r in trace if r['phase']=='refit']
  selected=max(inner,key=lambda r:r['inner_macro_f1'])
  check('epochs_selected_inner_only',selected['epoch']==m['frozen_epochs'])
  check('frozen_refit_epochs',len(refit)==m['frozen_epochs'])
  check('trace_no_outer_selection',all('outer' not in k for row in trace for k in row))
  check('total_budget',m['total_seconds']<900)
  p=PLEPreprocessor().fit(b.X.iloc[fit])
  check('fit_only_preprocessing_state',p.state()==json.loads((out/'preprocessing.json').read_text()))
  p_inner=PLEPreprocessor().fit(b.X.iloc[s['inner_fit']])
  check('inner_fit_only_preprocessing_state',p_inner.state()==json.loads((out/'inner_preprocessing.json').read_text()))
  network=PaperNeural(len(p.numeric_),p.category_sizes_,len(saved['classes']),1 if number==79 else 8)
  network.load_state_dict(torch.load(out/'model_state.pt',map_location='cpu',weights_only=True))
  infer=probabilities(network,tensors(p,b.X.iloc[valid]))
  check('saved_model_inference_probabilities',np.allclose(infer,prob,rtol=1e-6,atol=1e-7))
  check('saved_model_inference_labels',np.array_equal(saved['classes'][infer.argmax(1)],saved['prediction']))
  verification=dict(hypothesis=number,stage=stage,status='passed',checks=checks,passed=len(checks),failed=0,
                    note='No model fit; statistics recomputed from fit rows and saved weights used only for inference.')
  (out/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf8')
  print(json.dumps(dict(hypothesis=number,stage=stage,passed=len(checks),failed=0)),flush=True)
  all_results.append(verification)
print('TOTAL',sum(r['passed'] for r in all_results),flush=True)
