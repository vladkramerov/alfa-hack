"""Read-only instrumentation around an unchanged flat agent entrypoint.

No decisions, results, folds or API arguments are modified. Expensive predictions
and per-request accounting survive later failures; credentials are log-redacted.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path.cwd()))
import ag_budget  # Set the same process-wide start before loading the agent.
import numpy as np
import ag_llm_client as client
import ag_orchestrator as orchestrator
import ag_research_memory as memory
import ag_research_planner as planner

OUT = Path(os.environ['BENCH_RECEIPTS'])
OUT.mkdir(parents=True, exist_ok=True)
secret = os.environ.get('OPENAI_API_KEY', '')
factory = logging.getLogRecordFactory()


def redacted_record(*args, **kwargs):
    record = factory(*args, **kwargs)
    message = record.getMessage()
    record.msg = message.replace(secret, '[REDACTED]') if secret else message
    record.args = ()
    return record


logging.setLogRecordFactory(redacted_record)


def append(record):
    payload = json.dumps(record, ensure_ascii=False, default=str)
    assert not secret or secret not in payload
    with (OUT / 'events.jsonl').open('a') as f:
        f.write(payload + '\n')


def phase(system, user):
    if 'Сформулируй вывод' in user:
        return 'reflect'
    if 'Предложи следующую гипотезу' in user:
        return 'plan'
    if 'Функция make_features упала' in user:
        return 'repair'
    return 'understand'


request = client.LLMClient._request
chat_json = client.LLMClient.chat_json
load = orchestrator.Agent.load
add = memory.Memory.add
report = orchestrator.Agent.report
improves = orchestrator.Agent._is_improvement
llm_next = planner.Planner._llm_next


def request_hook(self, messages, max_tokens, temperature, timeout):
    started = time.monotonic()
    record = {'event': 'request', 'phase': phase(messages[0]['content'], messages[-1]['content']),
              'prompt_sha256': hashlib.sha256(json.dumps(messages).encode()).hexdigest(),
              'output_limit': max_tokens, 'timeout': timeout}
    try:
        result = request(self, messages, max_tokens, temperature, timeout)
        text, incoming, outgoing, finish = result
        record.update(ok=True, tokens_in=incoming, tokens_out=outgoing,
                      output_characters=len(text), finish_reason=finish)
        return result
    except Exception as exc:
        record.update(ok=False, error=type(exc).__name__)
        raise
    finally:
        record['seconds'] = time.monotonic() - started
        append(record)


def json_hook(self, system, user, *args, **kwargs):
    result = chat_json(self, system, user, *args, **kwargs)
    append({'event': 'json', 'phase': phase(system, user), 'valid': isinstance(result, dict)})
    return result


def load_hook(self):
    load(self)
    np.savez_compressed(OUT / 'splits.npz', A=self.A, B=self.B,
                        **{f'fit_{i}':a for i,(a,b) in enumerate(self.folds)},
                        **{f'valid_{i}':b for i,(a,b) in enumerate(self.folds)})
    (OUT / 'task.json').write_text(json.dumps(self.task.short(), ensure_ascii=False, default=str, indent=2))


def add_hook(self, result):
    add(self, result)
    if result.ok:
        np.savez_compressed(OUT / f'exp-{result.exp_id}.npz', oof=result.oof, test_pred=result.test_pred)
    append({'event':'experiment', **result.summary()})


def improves_hook(self, result, best):
    accepted = improves(self, result, best)
    append({'event':'decision', 'exp_id':result.exp_id, 'source':result.hypothesis.source,
            'accepted':bool(accepted), 'score':result.cv_mean if result.ok else None,
            'previous':best.cv_mean if best is not None else None})
    return accepted


def next_hook(self, *args, **kwargs):
    result = llm_next(self, *args, **kwargs)
    append({'event':'proposal', 'accepted':result is not None,
            'text':result.text if result is not None else None})
    return result


def report_hook(self, choice, single, ensemble):
    report(self, choice, single, ensemble)
    sol = choice.solution
    np.savez_compressed(OUT / 'final.npz', oof=sol.oof, test_pred=sol.test_pred)
    record = {'status':'completed','task':self.task.short(), 'metric':self.metric.name,
              'oof_score':sol.cv, 'holdout_B':sol.holdout_b, 'selected':sol.name,
              'llm':self.llm.stats(), 'stop_reason':self.stop_reason,
              'experiments':[r.summary() for r in self.memory.runs],
              'skipped':self.memory.skipped, 'seconds':self.budget.elapsed()}
    (OUT / 'agent.json').write_text(json.dumps(record,ensure_ascii=False,default=str,indent=2))


client.LLMClient._request = request_hook
client.LLMClient.chat_json = json_hook
orchestrator.Agent.load = load_hook
memory.Memory.add = add_hook
orchestrator.Agent._is_improvement = improves_hook
orchestrator.Agent.report = report_hook
planner.Planner._llm_next = next_hook
raise SystemExit(orchestrator.run(sys.argv[1:]))
