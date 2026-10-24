import sys,unittest,os
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import ag_finalize as F,ag_ml_metrics as M
from ag_schemas import RunResult,Hypothesis,PipelineSpec,TaskSpec
from ag_config import Config

class RefinementTests(unittest.TestCase):
 def setUp(self):
  self.rng=np.random.default_rng(917);self.A=np.arange(200);self.B=np.arange(200,400);self.metric=M.METRICS['rmse']
  M.configure(TaskSpec('regression','target','id','rmse',['id','target']))
  self.p=self.rng.normal(size=(400,2))*5+20
  self.runs=[RunResult(i,Hypothesis('fixture',PipelineSpec()),True,oof=self.p[:,i],test_pred=self.p[:7,i]) for i in range(2)]
  mean=self.p.mean(1);self.old=F.Solution('coarse',mean,mean[:7],{},0,0,{0:.5,1:.5})
 def go(self,y,callback=None):
  self.old.holdout_b=M.score(self.metric,y[self.B],self.old.oof[self.B])
  with patch.object(F,'_coarse_ensemble',return_value=self.old):
   if callback:
    with patch.object(F,'choose',side_effect=callback):return F.greedy_ensemble(self.metric,y,self.runs,self.A,self.B,8)
   return F.greedy_ensemble(self.metric,y,self.runs,self.A,self.B,8)
 def test_refines_existing_mixture_and_respects_simplex(self):
  y=self.p@np.array([.15,.85]);result=self.go(y)
  self.assertLess(result.holdout_b,self.old.holdout_b)
  self.assertAlmostEqual(sum(result.members.values()),1.)
  self.assertTrue(all(0<=v<=1 for v in result.members.values()))
  np.testing.assert_allclose(result.oof,sum(result.members[i]*r.oof for i,r in enumerate(self.runs)))
  np.testing.assert_allclose(result.test_pred,sum(result.members[i]*r.test_pred for i,r in enumerate(self.runs)))
 def test_rejects_refinement_that_hurts_B(self):
  y=self.p@np.array([.15,.85]);y[self.B]=self.p[self.B]@np.array([.85,.15])
  self.assertIs(self.go(y),self.old)
 def test_fitted_weights_do_not_depend_on_B_targets(self):
  saved=[]
  def capture(metric,y,reference,candidate,B,**kw):saved.append(candidate.members);return F.Choice(reference,'capture')
  y=self.p@np.array([.15,.85]);self.go(y,capture);y[self.B]+=1000;self.go(y,capture)
  self.assertEqual(saved[0],saved[1])
 def test_optimizer_failure_keeps_original(self):
  with patch('scipy.optimize.minimize',return_value=SimpleNamespace(success=False)):
   self.assertIs(self.go(self.p.mean(1)),self.old)
 def test_other_metrics_keep_original_behavior(self):
  with patch.object(F,'_coarse_ensemble',return_value=self.old):
   result=F.greedy_ensemble(M.METRICS['mae'],self.p.mean(1),self.runs,self.A,self.B,8)
  self.assertIs(result,self.old)
 def test_cpu_limit_applies_to_environment(self):
  for value,expected in [('1000',8),('0',1),('-10',1),('4',4)]:
   with patch.dict(os.environ,{'N_JOBS':value}):self.assertEqual(Config().n_jobs,expected)
if __name__=='__main__':unittest.main()
