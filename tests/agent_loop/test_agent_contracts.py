import unittest,tempfile,json
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
import pandas as pd
from ag_ml_dataset import Dataset
from ag_ml_pipeline import build_matrices
from ag_schemas import PipelineSpec,Hypothesis,RunResult,TaskSpec
from ag_research_memory import Memory
from ag_research_planner import Planner
from ag_ml_metrics import METRICS
from ag_understand_profiler import compact
from ag_llm_client import LLMClient
from ag_io_contract import LLMSettings
from ag_config import Config

class Contracts(unittest.TestCase):
 def test_raw_dropped_operand_survives_until_code(self):
  X=pd.DataFrame({'raw':[2.,3.,4.,5.],'other':[1.,2.,4.,8.]})
  ds=Dataset(X.iloc[:3].copy(),X.iloc[3:].copy(),np.arange(3),None,[])
  spec=PipelineSpec(drop_columns=['raw'],feature_code='def make_features(df):\n    df = df.copy()\n    df["ratio"] = df["raw"] / (df["other"] + 1)\n    return df')
  a,b,_=build_matrices(ds,spec)
  self.assertNotIn('raw',a);np.testing.assert_allclose(a['ratio'],[1,1,.8])
  self.assertIn('raw',ds.X_train);self.assertNotIn('ratio',ds.X_train)
 def test_datetime_raw_input_before_expansion(self):
  X=pd.DataFrame({'date':['2020-01-01','2021-01-01','2022-01-01'],'x':[1.,2.,3.]})
  ds=Dataset(X.iloc[:2],X.iloc[2:],np.arange(2),None,['date'])
  spec=PipelineSpec(feature_code='def make_features(df):\n    df = df.copy()\n    df["year_from_raw"] = pd.to_datetime(df["date"]).dt.year\n    return df')
  a,_,_=build_matrices(ds,spec);np.testing.assert_allclose(a['year_from_raw'],[2020,2021])
 def test_profile_schema_includes_less_important_semantics(self):
  p={'columns':[{'name':f'x{i}','dtype':'float','desc':'meaning','null':0} for i in range(50)]}
  c=compact(p,3,important=['x49']);self.assertEqual(c['columns'][0]['name'],'x49')
  self.assertEqual(len(c['schema']),50);self.assertIn('meaning',c['schema']['x30']);self.assertEqual(len(p['columns']),50)
 def test_memory_deltas_and_incumbent_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   m=Memory(METRICS['rmse'],Path(d)/'memory.jsonl')
   for i in range(10):m.add(RunResult(i,Hypothesis('test',PipelineSpec(params={'a':i},drop_columns=['x']*10)),True,cv_mean=10-i))
   m.incumbent=m.runs[0];before=[r.hypothesis.spec.key() for r in m.runs]
   rows=m.prompt_summary();self.assertTrue(any(r['exp_id']==0 and 'spec' in r for r in rows))
   for r in rows:
    if r['exp_id']:
     reconstructed={**m.incumbent.hypothesis.spec.to_dict(),**r['vs_best']}
     self.assertEqual(reconstructed,m.runs[r['exp_id']].hypothesis.spec.to_dict())
   self.assertEqual(before,[r.hypothesis.spec.key() for r in m.runs])
 def test_explicit_endpoint_key_model_required(self):
  for setting in [LLMSettings(None,'fake','model'),LLMSettings('https://example.invalid',None,'model'),LLMSettings('https://example.invalid','fake',None)]:
   c=LLMClient(setting,Config());self.assertFalse(c.available())
 def test_context_budget_prevents_unfunded_call(self):
  c=LLMClient(LLMSettings('https://example.invalid','fake','model'),Config(llm_max_tokens_total=100))
  c._request=Mock();self.assertIsNone(c.chat('system','long'*100,max_tokens=64));c._request.assert_not_called()
 def test_valid_budget_and_json(self):
  c=LLMClient(LLMSettings('https://example.invalid','fake','model'),Config(llm_max_tokens_total=1000))
  c._request=Mock(return_value=('{"ok":true}',20,5,'stop'))
  self.assertEqual(c.chat_json('s','u',100),{'ok':True});self.assertEqual(c.tokens_in+c.tokens_out,25)
 def test_slow_warmup_allows_llm_after_structural_probes(self):
  task=TaskSpec('regression','y','id','rmse',['id','y']);llm=Mock();llm.available.return_value=True
  fallback=Mock();p=Planner(task,{},METRICS['rmse'],llm,fallback,[],warmup_fallback=12)
  p._llm_next=Mock(return_value='hypothesis');memory=Mock();memory.runs=[None]*4
  self.assertEqual(p.next(memory,150,80),'hypothesis');fallback.next.assert_not_called()
  p._llm_next.reset_mock();memory.runs=[None]*3;p.next(memory,150,80);p._llm_next.assert_not_called()

class NoOpContracts(unittest.TestCase):
 def agent(self, values):
  from ag_orchestrator import Agent
  a=Agent.__new__(Agent)
  X=pd.DataFrame({'x':values,'z':np.arange(len(values),dtype=float)})
  a.ds=Dataset(X.iloc[:-1].copy(),X.iloc[-1:].copy(),np.arange(len(values)-1),None,[])
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  a.memory=Memory(METRICS['rmse'],Path(self.tmp.name)/'memory.jsonl')
  a.memory.add(RunResult(0,Hypothesis('base',PipelineSpec()),True,cv_mean=1))
  return a
 def test_exact_no_op_skips_fit(self):
  a=self.agent(np.arange(1.,101.));h=Hypothesis('cleanup',PipelineSpec(features=['placeholders_as_missing']))
  self.assertTrue(a._unchanged_features(h));self.assertTrue(a.memory.tried(h.spec.key()))
 def test_effective_cleanup_kept(self):
  a=self.agent(np.r_[0.,np.arange(1950.,2049.)]);h=Hypothesis('cleanup',PipelineSpec(features=['placeholders_as_missing']))
  self.assertFalse(a._unchanged_features(h))
 def test_model_change_and_supervised_change_kept(self):
  a=self.agent(np.arange(1.,101.))
  self.assertFalse(a._unchanged_features(Hypothesis('model',PipelineSpec(params={'num_leaves':20},features=['placeholders_as_missing']))))
  self.assertFalse(a._unchanged_features(Hypothesis('supervised',PipelineSpec(features=['row_memory']))))

class RecoveryContracts(unittest.TestCase):
 def test_isolated_error_and_timeout_leave_checkpoint(self):
  import time
  from ag_exec_runner import run_with_timeout,ExperimentError,ExperimentTimeout
  with tempfile.TemporaryDirectory() as d:
   checkpoint=Path(d)/'submission.csv';checkpoint.write_text('id,target\n1,2\n');before=checkpoint.read_bytes()
   with self.assertRaises(ExperimentError):run_with_timeout(lambda:1/0,(),1)
   with self.assertRaises(ExperimentTimeout):run_with_timeout(time.sleep,(1,),.02)
   self.assertEqual(checkpoint.read_bytes(),before)
 def test_json_from_reasoning_and_fences(self):
  from ag_llm_client import extract_json
  self.assertEqual(extract_json('<think>draft</think>```json\n{"change":{}}\n```'),{'change':{}})
  self.assertIsNone(extract_json('<think>unfinished reasoning'))

class TemporalPolicyContracts(unittest.TestCase):
 def test_load_maps_latest_fold_indices_and_keeps_external_target_absent(self):
  from ag_io_contract import resolve_task
  from ag_orchestrator import Agent
  from ag_budget import Budget
  import time
  rng=np.random.default_rng(91)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);n=5000
   tr=pd.DataFrame({'id':np.arange(n),'period':np.repeat(np.arange(5),1000),'x':rng.normal(size=n),'target':rng.normal(size=n)})
   te=pd.DataFrame({'id':np.arange(n,n+300),'period':5,'x':rng.normal(size=300)})
   tr.to_csv(p/'train.csv',index=False);te.to_csv(p/'test.csv',index=False)
   pd.DataFrame({'id':te.id,'target':0.}).to_csv(p/'sample_submission.csv',index=False)
   (p/'task.md').write_text('Primary metric: rmse\nTarget: target\nIdentifier: id\nTraining data: train.csv\nTest data: test.csv\nSample submission: sample_submission.csv\n')
   cfg=Config(llm_enabled=False);a=Agent(resolve_task([str(p/'task.md')]),cfg,Budget(cfg,start=time.time()));a.load()
   self.assertEqual(a.structure.get('selection_scope'),'latest_temporal_fold')
   np.testing.assert_array_equal(np.sort(np.r_[a.A,a.B]),np.sort(a.folds[-1][1]))
   self.assertEqual(len(np.intersect1d(a.A,a.B)),0)
   for fit,valid in a.folds:
    self.assertLess(tr.period.iloc[fit].max(),tr.period.iloc[valid].min())
   self.assertNotIn('target',a.ds.X_train);self.assertNotIn('target',a.ds.X_test)
   self.assertTrue((p/'submission.csv').exists())

class DuplicateValidationContracts(unittest.TestCase):
 def test_unseen_duplicate_groups_never_cross_folds(self):
  from ag_ml_validation import unseen_duplicate_folds
  task=TaskSpec('multiclass','y','id','macro_f1',['id','y'],classes=[0,1])
  x=pd.DataFrame({'a':np.repeat(np.arange(100),3)});y=np.repeat(np.arange(100)%2,3)
  result=unseen_duplicate_folds(task,x,pd.DataFrame({'a':np.arange(200,220)}),y,5,42)
  self.assertIsNotNone(result)
  for tr,va in result[0]:
   self.assertFalse(set(x.a.iloc[tr]) & set(x.a.iloc[va]));self.assertFalse(set(tr)&set(va))
  self.assertIsNone(unseen_duplicate_folds(task,x,x.iloc[:20],y,5,42))
 def test_unique_rows_keep_original_policy(self):
  from ag_ml_validation import unseen_duplicate_folds
  task=TaskSpec('multiclass','y','id','macro_f1',['id','y'],classes=[0,1]);x=pd.DataFrame({'a':np.arange(100)})
  self.assertIsNone(unseen_duplicate_folds(task,x,pd.DataFrame({'a':[200]}),np.arange(100)%2,5,42))

if __name__=='__main__':unittest.main()
