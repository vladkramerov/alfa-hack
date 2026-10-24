from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.promote_sprint import paired_write,current_coverage


class PromotionTests(unittest.TestCase):
    def test_stale_verification_or_running_attempt_refused(self):
        import json
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder:
            suite=Path(folder)/'suite';hyp=suite/'HYP-104';hyp.mkdir(parents=True)
            path=hyp/'attempts.json'
            path.write_text(json.dumps([{'directory':'results/h104','status':'started'}]))
            with self.assertRaises(AssertionError):current_coverage(suite,{'runs':[]})
            path.write_text(json.dumps([{'directory':'results/h104','status':'completed'}]))
            with self.assertRaises(AssertionError):current_coverage(suite,{'runs':[]})
            current_coverage(suite,{'runs':[{'directory':'results/h104'}]})

    def test_pair_restored_on_second_replace_failure(self):
        import os
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder:
            root=Path(folder);a=root/'baseline.json';b=root/'known.py';backup=root/'backups';backup.mkdir()
            a.write_bytes(b'old baseline');b.write_bytes(b'old known')
            original=os.replace;calls=[]
            def fail_second(src,dst):
                calls.append(str(dst))
                if len(calls)==2:raise OSError('simulated disk error')
                return original(src,dst)
            with patch('experiments.promote_sprint.os.replace',side_effect=fail_second):
                with self.assertRaises(OSError):paired_write(a,b,{'new':1},'new known',backup)
            self.assertEqual(a.read_bytes(),b'old baseline');self.assertEqual(b.read_bytes(),b'old known')
            self.assertEqual((backup/'automl-known-before-promotion.py').read_bytes(),b'old known')
            self.assertFalse(list(root.glob('*.promotion-tmp')))

    def test_pair_success_and_backups_preserved(self):
        import json
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder:
            root=Path(folder);a=root/'baseline.json';b=root/'known.py';backup=root/'backups';backup.mkdir()
            a.write_bytes(b'old baseline');b.write_bytes(b'old known')
            paired_write(a,b,{'new':1},'new known',backup)
            self.assertEqual(json.loads(a.read_text()),{'new':1});self.assertEqual(b.read_text(),'new known')
            self.assertEqual((backup/'baseline-before-promotion.json').read_bytes(),b'old baseline')
