import unittest,tempfile
from pathlib import Path
class AlternativeRunnerTests(unittest.TestCase):
    def test_frozen_wave_has_ten_cases_and_profile_memory_first(self):
        from experiments.run_alternatives import experiment_plan,BASELINE
        jobs=experiment_plan();self.assertEqual(len(jobs),10);self.assertEqual(jobs[0]['number'],69)
        self.assertEqual(sum(j['task']=='multiclass' for j in jobs),4)
        self.assertEqual(BASELINE['regression'],40.972997661684175)
        self.assertEqual(BASELINE['multiclass'],.6495311620074445)
    def test_attempt_claim_survives_rename(self):
        from experiments.run_alternatives import experiment_plan,claim
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as d:
            p=Path(d);(p/'HYPOTHESES.md').write_text('HYP-069',encoding='utf8');j=experiment_plan()[0]
            self.assertEqual(claim(p,j,{'train':'x'},'split')['status'],'started')
            with self.assertRaises(RuntimeError):claim(p,{**j,'description':'renamed'},{'train':'x'},'split')
