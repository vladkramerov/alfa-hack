import os,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

class Toy:
    def fit(self,X,y): self.rows=len(X); return self

class ComboMemoryTests(unittest.TestCase):
    def test_attempt_is_persistent_and_renamed_seed_cannot_repeat(self):
        from automl_combo_memory import fit_once
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as d,patch.dict(os.environ,{'COMBO_RUN_DIR':d,'COMBO_LEDGER_PATH':str(root/'HYPOTHESES.md')}):
            X=pd.DataFrame({'x':[1.,2.,3.]}); y=np.arange(3.)
            self.assertEqual(fit_once(Toy(),X,y,{'family':'toy','random_state':42}).rows,3)
            with self.assertRaises(RuntimeError):fit_once(Toy(),X,y,{'family':'toy','random_state':12,'hypothesis':'another name'})
            a=json.loads((Path(d)/'model-fit-attempts.jsonl').read_text().strip())
            self.assertEqual(a['status'],'started');self.assertEqual(len(a['ledger_sha256']),64)
    def test_changed_actual_features_are_new(self):
        from automl_combo_memory import fit_fingerprint
        X=pd.DataFrame({'x':[1.,2.]});y=np.arange(2.)
        self.assertNotEqual(fit_fingerprint(X,y,{'family':'toy'}),fit_fingerprint(X.assign(new=0.),y,{'family':'toy'}))

if __name__=='__main__':unittest.main()
