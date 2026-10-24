# HYP-028 regression: техническое продолжение

Предыдущий запуск остановлен до training новой outer-модели чрезмерно строгой защитой. Его ошибка и артефакты сохранены. Новая гипотеза не придумана, прежние модели не повторяются.

[Доказательство нового набора признаков](continuation-proof.json); все inner-предсказания/модели и permutation importance загружаются из кэша. [Исходники](../series-twenty-hypotheses-v1/continuations/source/), [evidence](evidence.json).

Команда: `.venv-ml/Scripts/python.exe -X utf8 -u -m experiments.continue_twenty_blocked_selection`.

Итог: completed, rmse=42.245303736669484; Не подтверждена по заданному критерию. [Метрики](metrics.json), [аудит](preprocessing-audit.json), [предсказания](validation_predictions.npz). Внешнийholdoutужеиспользовался; выводразведочный.
