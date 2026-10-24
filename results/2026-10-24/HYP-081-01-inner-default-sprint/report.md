# HYP-081/inner/default

Started before fit 2026-10-24T02:11:48.348165+03:00. [SPEC011](../../../docs/specs/SPEC-011-three-hour-improvement.md).

No old estimator refits. Fixed source/parameters; exploratory research, not private score.

Итог: completed; rmse=30.689409165993794; reference=42.46005247026025. [Метрики](metrics.json), [аудит](audit.json), [fit claim](../series-three-hours-v1/HYP-081/model-fit-history.json). Повтор не запускать.
Дополнение: сопоставимый incremental контроль H70inner30.663255379972725; H81=30.689409165993794 хуже на0.026153786. По сравнению с H69 улучшение унаследовано от H70. Новое fuzzy-сопоставление не подтвердилось, outer не запускать.

Датированное дополнение к provenance: Команда запуска: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.sprint_run --number 81 --stage inner --variant default`; cwd=`C:\Users\User\PycharmProjects\Alpha_Hackathon`. Exact frozen worker source/hash перечислены в evidence.json и suite/HYP-081/frozen.json; inputs/dataSHA=data.json, split=splits.npz, Python3.12 ML environment .venv-ml (providedrequirements unchanged). Старые контрольные fit=0 по сохранённым claims; originalouter exploratory.
