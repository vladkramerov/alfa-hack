"""H65 regression contracts using the repaired shared output codec.

The original diagnostic controller/evidence is immutable; these tests adapt its
CSV read boundary to the same read_submission API used by the application.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.check_schema_semantics import run_case,CASE_NAMES
from automl_data import make_submission,atomic_submission,read_submission


def _typed_roundtrip(bundle,prediction):
    frame=make_submission(bundle,prediction)
    destination=bundle.task_path.parent/'submission.csv'
    atomic_submission(bundle,frame,destination)
    return frame,read_submission(bundle,destination)


class SchemaSemanticsTests(unittest.TestCase):
    def check_case(self,case):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder, \
                patch('experiments.check_schema_semantics._write',_typed_roundtrip):
            run_case(case,folder)


def _test(case):
    def test(self):self.check_case(case)
    return test


for _case,_name in CASE_NAMES.items():
    setattr(SchemaSemanticsTests,'test_'+_case+'_'+_name,_test(_case))


if __name__=='__main__':unittest.main()
