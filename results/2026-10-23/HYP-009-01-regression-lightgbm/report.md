# HYP-009: regression / lightgbm

[Спецификация](../../../docs/specs/SPEC-003-platform-agent.md), [гипотеза](../../../docs/hypotheses/HYP-009.md).

Новая модель; известные контрольные конфигурации повторно не обучались. Разбиение побайтно совпало с сохранённым outer. Предобработка: train-only median, native categories (LightGBM) / one-hot (ExtraTrees).

Команда: `.venv-ml/Scripts/python.exe -m experiments.run_new_families`.

Статус: completed; метрика: rmse; значение: 60.492688256313926; секунд: 1.7660000000032596.

[Хеши данных/кода, окружение и конфигурация](evidence.json), [метрика](metrics.json), [лог](worker.log). Повторно используемый validation: исследовательский выбор без независимого подтверждения. Вывод и сравнение: общий NEW_EXPERIMENT_REPORT.md.
