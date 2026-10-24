"""No dataset-specific models, labels, scores or weights ship with the agent.

Offline benchmark evidence is retained in results, outside the platform ZIP.
Each task learns its own calibration and keeps its own local hypothesis ledger.
"""
KNOWN_TASKS = []
