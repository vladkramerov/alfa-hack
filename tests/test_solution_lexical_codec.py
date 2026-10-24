"""Exercise both solution-level CSV boundaries without fitting any model."""
import os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from automl_data import atomic_submission,make_submission,read_submission
from tests import test_csv_codec as codec_fixtures

class SolutionLexicalCodecTests(unittest.TestCase):
    def test_evaluate_refit_and_final_read_preserve_original_strings(self):
        import solution
        calls=[]
        def fake_worker(bundle,candidate,mode,directory,splits,timeout):
            directory.mkdir(parents=True);calls.append(mode)
            if mode=='evaluate':return {'status':'completed','metric':'macro_f1','score':.9}
            labels=np.resize(np.array(['001','010']),len(bundle.test))
            atomic_submission(bundle,make_submission(bundle,labels),directory/'candidate_submission.csv')
            return {'status':'completed','train_rows':len(bundle.train),'test_rows':len(bundle.test)}
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as d:
            bundle=codec_fixtures.CsvCodecTests().fixture(d,['0007','0012'],['001','010'])
            with patch.dict(os.environ,{'AUTOML_MAX_EXPERIMENTS':'1','AUTOML_PORTFOLIO':'1','AUTOML_TIME_LIMIT':'300'}),patch.object(solution,'worker',side_effect=fake_worker):
                solution.main(bundle.task_path)
            frame=read_submission(bundle,Path(d)/'submission.csv')
            self.assertEqual(calls,['evaluate','refit'])
            self.assertEqual(frame.key.tolist(),['0007','0012']);self.assertEqual(frame.response.tolist(),['001','010'])

if __name__=='__main__':unittest.main()
