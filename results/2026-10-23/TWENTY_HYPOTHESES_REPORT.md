# Двадцать новых гипотез HYP-022…041 — результаты

Всего 25 попыток / 20 гипотез; оценок с метриками 23, технических остановок 2; случаев без завершённой метрики 0.
Независимая сверка passed: 539 успешных / 0 ошибок. Обучение при сверке не запускалось.

| Гипотеза | Задача | Метрика | Значение | Прежний лучший | Улучшение | Исход |
| --- | --- | --- | ---: | ---: | ---: | --- |
| [HYP-022](HYP-022-01-regression-spec006/report.md) | regression | rmse | 42.203267905 | 42.196700027 | -0.006567878 | completed |
| [HYP-022](HYP-022-02-multiclass-spec006/report.md) | multiclass | macro_f1 | 0.561552536 | 0.591451371 | -0.029898835 | completed |
| [HYP-023](HYP-023-01-regression-spec006/report.md) | regression | rmse | 43.192109473 | 42.196700027 | -0.995409446 | completed |
| [HYP-024](HYP-024-01-regression-spec006/report.md) | regression | rmse | 42.183546284 | 42.196700027 | +0.013153744 | completed |
| [HYP-025](HYP-025-01-multiclass-spec006/report.md) | multiclass | macro_f1 | 0.577172631 | 0.591451371 | -0.014278739 | completed |
| [HYP-026](HYP-026-01-multiclass-spec006/report.md) | multiclass | macro_f1 | 0.579398705 | 0.591451371 | -0.012052666 | completed |
| [HYP-027](HYP-027-01-multiclass-spec006/report.md) | multiclass | macro_f1 | 0.585738591 | 0.591451371 | -0.005712780 | completed |
| [HYP-028](HYP-028-01-regression-spec006/report.md) | regression | None | — | 42.196700027 | — | failed |
| [HYP-028](HYP-028-02-multiclass-spec006/report.md) | multiclass | None | — | 0.591451371 | — | failed |
| [HYP-029](HYP-029-01-regression-spec006/report.md) | regression | rmse | 42.086102267 | 42.196700027 | +0.110597760 | completed |
| [HYP-030](HYP-030-01-regression-spec006/report.md) | regression | rmse | 41.993466003 | 42.196700027 | +0.203234024 | completed |
| [HYP-031](HYP-031-01-regression-spec006/report.md) | regression | rmse | 42.004603937 | 42.196700027 | +0.192096090 | completed |
| [HYP-032](HYP-032-01-regression-spec006/report.md) | regression | rmse | 43.810373478 | 42.196700027 | -1.613673450 | completed |
| [HYP-033](HYP-033-01-regression-spec006/report.md) | regression | rmse | 42.974266299 | 42.196700027 | -0.777566271 | completed |
| [HYP-034](HYP-034-01-regression-spec006/report.md) | regression | rmse | 44.423934955 | 42.196700027 | -2.227234928 | completed |
| [HYP-035](HYP-035-01-regression-spec006/report.md) | regression | rmse | 41.480963028 | 42.196700027 | +0.715736999 | completed |
| [HYP-036](HYP-036-01-regression-spec006/report.md) | regression | rmse | 42.255091314 | 42.196700027 | -0.058391287 | completed |
| [HYP-037](HYP-037-01-multiclass-spec006/report.md) | multiclass | macro_f1 | 0.628604790 | 0.591451371 | +0.037153419 | completed |
| [HYP-038](HYP-038-01-multiclass-spec006/report.md) | multiclass | macro_f1 | 0.595178021 | 0.591451371 | +0.003726650 | completed |
| [HYP-039](HYP-039-01-multiclass-spec006/report.md) | multiclass | macro_f1 | 0.579583443 | 0.591451371 | -0.011867928 | completed |
| [HYP-040](HYP-040-01-multiclass-spec006/report.md) | multiclass | macro_f1 | 0.592240246 | 0.591451371 | +0.000788875 | completed |
| [HYP-041](HYP-041-01-regression-spec006/report.md) | regression | rmse | 42.196700027 | 42.196700027 | +0.000000000 | completed |
| [HYP-041](HYP-041-02-multiclass-spec006/report.md) | multiclass | macro_f1 | 0.591451371 | 0.591451371 | +0.000000000 | completed |
| [HYP-028](HYP-028-03-regression-guard-continuation/report.md) | regression | rmse | 42.245303737 | 42.196700027 | -0.048603709 | completed |
| [HYP-028](HYP-028-04-multiclass-guard-continuation/report.md) | multiclass | macro_f1 | 0.567615694 | 0.591451371 | -0.023835677 | completed |

Улучшение положительно для обеих метрик: прежний−новый RMSE, новый−прежний F1.

Лучший новый regression: **HYP-035, rmse=41.480963027907**, улучшение +0.715736999287.
Лучший новый multiclass: **HYP-037, macro_f1=0.628604789914**, улучшение +0.037153418995.

Восстановление только относительно full-v1 (RMSE43.837201829105254 / F1.5637871673383791) не означает новый рекорд.
Преобразования и imputation обучаются только на соответствующем training subset (в итоговой модели на outer-fit); learned selection/thresholds/коррекции выбираются на inner. Веса H36 — фиксированная формула по training-годам. Исходный outer уже использовался и не является независимым подтверждением. Реальный платформенный score не оценивался. Точный эффект отдельных шагов старого HYP-021 здесь не устанавливается.
Два первоначальных HYP-028 могли остановиться до нового outer-fit из-за чрезмерно строгого guard. Технические продолжения используют готовые inner-модели и permutation importance; исходные ошибки сохранены, новыми гипотезами или повторными baseline fits не считаются.
HYP-024 включает групповой imputer и missing indicators, HYP-022 меняет и сохранение строк, и clipping; эффект отдельного элемента этих составных представлений не установлен.

История/карточки/results обновлены после каждого запуска, отрицательные исходы сохранены. Старые outer-baseline predictions переиспользовались; модельные fits новой серии и внутренние cross-fit отражены в журнале model-fit-attempts.jsonl.

[Независимая сверка](series-twenty-hypotheses-v1/independent-verification.json), [CSV](twenty-hypotheses-comparison.csv), [план до fit](series-twenty-hypotheses-v1/frozen-plan.json), [реестр](series-twenty-hypotheses-v1/records.json).
