import json,tempfile,unittest
from pathlib import Path

class CombinationRunnerTests(unittest.TestCase):
    def test_seven_new_graphs_use_current_incumbents(self):
        from experiments.run_combinations import experiment_plan,BASELINE
        plan=experiment_plan()
        self.assertEqual([p['number'] for p in plan],list(range(42,49)))
        self.assertEqual(sum(p['task']=='regression' for p in plan),4)
        self.assertEqual(BASELINE['regression'],41.480963027907194)
        self.assertEqual(BASELINE['multiclass'],.6286047899141216)
    def test_started_claim_cannot_be_replayed_after_rename(self):
        from experiments.run_combinations import claim_attempt,experiment_plan
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as d:
            p=Path(d);(p/'HYPOTHESES.md').write_text('HYP-042 authorized',encoding='utf-8')
            job=experiment_plan()[0]
            record=claim_attempt(p,job,{'train':'abc'},'split')
            self.assertEqual(record['status'],'started')
            with self.assertRaises(RuntimeError):claim_attempt(p,{**job,'description':'renamed'}, {'train':'abc'},'split')

if __name__=='__main__':unittest.main()
