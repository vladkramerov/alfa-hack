"""Fresh frozen team delivery on the existing masked CLI fixtures; resumable receipts."""
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_data import load_task, validation_split, split_key, score
from automl_runtime import child_environment

OUT = ROOT / 'results/team-runtime'
CLASS_ONLY = len(sys.argv) > 1 and sys.argv[1] == '--classification-official-env'
WORK = Path('/private/tmp/alfa-team-runtime-control' + ('-official-env' if CLASS_ONLY else ''))
REF = 'origin/ivan_branch'
PREFIX = 'deliveries/sergey_ivan_fusion_v1/'


def main():
    OUT.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    receipt = OUT / ('control-classification-official-env.json' if CLASS_ONLY else 'control.json')
    commit = subprocess.check_output(['git', 'rev-parse', REF], text=True).strip()
    report = json.loads(receipt.read_text()) if receipt.exists() else {
        'source_commit': commit, 'source_prefix': PREFIX, 'runs': [],
        'protocol': 'Fresh whole CLI, same masked input CSV bytes as Maxim f3a ZIP; 8CPU ceiling,1200seconds,cap16,offline. Team worker internally uses4threads. Repeatedly viewed outer, not independent evidence.',
        'status': 'running'}
    assert report['source_commit'] == commit
    def save():
        receipt.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    save()
    for task, folder in [('regression', 'Регрессия'), ('classification', 'Многоклассовая классификация')]:
        if CLASS_ONLY and task != 'classification':
            continue
        if any(r['task'] == task for r in report['runs']):
            continue
        directory = WORK / task
        directory.mkdir(exist_ok=True)
        fixture = Path('/private/tmp/alfa-strongest-e2e-v2') / (task + '-shadow')
        source_files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', REF, PREFIX], text=True).splitlines()
        hashes = {}
        for name in source_files:
            relative = name.removeprefix(PREFIX)
            assert '/' not in relative
            content = subprocess.check_output(['git', 'show', REF + ':' + name])
            (directory / relative).write_bytes(content)
            hashes[relative] = hashlib.sha256(content).hexdigest()
        for name in ['task.md', 'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv']:
            (directory / name).write_bytes((fixture / name).read_bytes())
        bundle = load_task(ROOT / 'drive-download-20261023T201455Z-1-001' / folder / 'task.md')
        fit, valid, _ = validation_split(bundle)
        env = child_environment(8)
        env.update(AUTOML_TIME_LIMIT='1200', AUTOML_MAX_EXPERIMENTS='16', AUTOML_THREADS='8')
        # Explicitly offline: never send task contents or keys in this comparison.
        for key in ['OPENAI_BASE_URL', 'OPENAI_MODEL', 'OPENAI_API_KEY']:
            env.pop(key, None)
        summaries = list(directory.glob('agent_runs/*/summary.json'))
        if summaries:
            raise RuntimeError('Existing completed run missing receipt: inspect before any restart')
        if (directory / 'process.json').exists():
            raise RuntimeError('Existing process receipt: inspect before any restart')
        started = time.monotonic()
        with (directory / 'cli.log').open('w') as log:
            child = subprocess.Popen([sys.executable, 'solution.py', 'task.md'], cwd=directory,
                                     env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (directory / 'process.json').write_text(json.dumps({'pid': child.pid, 'created': time.time()}))
        process = psutil.Process(child.pid)
        peak = 0
        while child.poll() is None:
            try:
                peak = max(peak, sum(p.memory_info().rss for p in [process] + process.children(recursive=True)) / 1024**2)
            except psutil.Error:
                pass
            if time.monotonic() - started > 1240 or peak > 32000:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                break
            time.sleep(.5)
        summaries = list(directory.glob('agent_runs/*/summary.json'))
        result = {'task': task, 'exit_code': child.returncode, 'seconds': time.monotonic()-started,
                  'peak_rss_mb': peak, 'source_hashes': hashes, 'split_key': split_key(fit, valid)}
        if child.returncode == 0 and summaries:
            summary = json.loads(summaries[-1].read_text())
            submission = pd.read_csv(directory / 'submission.csv')
            sample = pd.read_csv(directory / 'sample_submission.csv')
            assert list(sample) == list(submission) and len(submission) == len(valid)
            assert not submission.isna().any().any()
            assert submission[bundle.id_column].equals(sample[bundle.id_column])
            result.update(score=score(bundle.metric, bundle.y.iloc[valid], submission[bundle.target].to_numpy(), classes=bundle.classes),
                          selected=summary['selected'], summary=str(summaries[-1]),
                          submission_sha256=hashlib.sha256((directory / 'submission.csv').read_bytes()).hexdigest())
        report['runs'].append(result)
        save()
        print(json.dumps(result, ensure_ascii=False), flush=True)
    report['status'] = 'completed'
    save()


if __name__ == '__main__':
    main()
