# SPEC-008: семь новых комбинаций HYP-042–048

Независимая сверка: **passed**; успешных проверок 63, ошибок 0.
Завершённых оценок 7/7; terminal outcomes 7/7; новых модельных fit 24.
Сверка не запускала обучение. Проверены сохранённые исходники против frozen SHA, исходные targets, split/order, OOF и журнал fit.

| Гипотеза | Задача | Статус | Метрика | Пересчитано | Incumbent | Улучшение | Проверено |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| HYP-042 | regression | completed | rmse | 41.504841314734 | 41.480963027907 | -0.023878286827 | True |
| HYP-043 | regression | completed | rmse | 40.972997661684 | 41.480963027907 | +0.507965366223 | True |
| HYP-044 | regression | completed | rmse | 41.944161160505 | 41.480963027907 | -0.463198132598 | True |
| HYP-045 | regression | completed | rmse | 41.406800898690 | 41.480963027907 | +0.074162129217 | True |
| HYP-046 | multiclass | completed | macro_f1 | 0.615435983346 | 0.628604789914 | -0.013168806568 | True |
| HYP-047 | multiclass | completed | macro_f1 | 0.633656011199 | 0.628604789914 | +0.005051221285 | True |
| HYP-048 | multiclass | completed | macro_f1 | 0.649531162007 | 0.628604789914 | +0.020926372093 | True |

Положительное улучшение: incumbent−новый RMSE; новый F1−incumbent. Строгий порог допуска >1e-10.

К переносу после верификации: **HYP-043 / regression = 40.972997661684**; frozen candidate и learned state сохранены в comparison.json.
К переносу после верификации: **HYP-048 / multiclass = 0.649531162007**; frozen candidate и learned state сохранены в comparison.json.

Это исследовательские сравнения на ранее просмотренном outer holdout; статистическая значимость и hidden platform score не установлены.
Отрицательные исходы и технические ошибки сохранены. Незавершённая проверка не считается опровержением гипотезы.

[Все проверки](verification.json), [сравнение и frozen winners](comparison.json), [план до fit](frozen-plan.json).
