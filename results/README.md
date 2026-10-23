# Результаты проверок

Новая поставка SPEC009: [solution_alternative_v2.zip](../dist/solution_alternative_v2.zip),61 330байт/16rootfiles; RMSE39.392194 и MacroF1.649531. [9численныхоценок+1техническийисход](2026-10-24/series-alternatives-v1/ALTERNATIVES_REPORT.md),65independentchecks/22newfits;206units. [H65CSVисправление](2026-10-24/HYP-065-02-schema-codec-fix/verification.json), [H68неполнаядиагностикаscore](2026-10-24/HYP-068-01-platform-provenance/report.md). СтарыеZIP/негативныеотчётысохранены, hidden scoreнеизмерен.

Актуальная поставка SPEC-008: [solution_combined_metrics.zip](../dist/solution_combined_metrics.zip),57110байт; H43RMSE40.972998/H48Macro-F1.649531. [Семь сравнений](2026-10-23/series-combinations-v1/COMBINATIONS_REPORT.md),63проверки/24новыхfitбезповторов; [baseline](../baseline.json). Состояния приёмки ZIP фиксируются отдельными отчётами, скрытый score не измерен. Крупные OOFcache/модели/синтетические данные сохраняются локально и исключены из Git.

Текущая поставка: [улучшенный ZIP](../dist/solution_improved_metrics.zip), H35RMSE41.480963/H37Macro-F1.628605; [сверка ZIP](2026-10-23/IMPROVED_DELIVERY.json). Все20новыхгипотеззавершены: [23метрикии2техническиеостановки](2026-10-23/TWENTY_HYPOTHESES_REPORT.md),539checksOK.126unitтестов,5mockсценариев,2fullrefit; повторныхvalidationfitнет.

Предыдущая поставка: [10 новых гипотез/11оценок](2026-10-23/TEN_HYPOTHESES_REPORT.md), затем [обязательная full-v1/2оценки](2026-10-23/FULL_PREPROCESSING_REPORT.md). Составная полная обработка ухудшила обе локальные метрики; нормализация выполнена по прямому требованию пользователя. [Итоговый новый ZIP](../dist/solution_full_preprocessing.zip), [75 тестов](2026-10-23/full-preprocessing-final-tests.log).

[Итоговый отчёт](2026-10-23/EXPERIMENT_REPORT.md) и [CSV всех оценок](2026-10-23/comparison.csv) доступны после завершения серии.

SPEC-002: два датасета, inner для выбора, заранее замороженная outer-панель. Для классификации одинаковые исходные строки сгруппированы; прежние random-split метрики напрямую не сравниваются.
Каждая папка запуска содержит report.md, metrics.json, preprocessing audit и validation predictions. Полное обучение — технический этап без новой метрики. Экспорт полного запрошенного preprocessing хранится отдельно от выбранной модели.

Повторный запуск не перезаписывает историю; правила структуры и [шаблон](_template.md) сохраняются.

## Журнал

| Начало (Москва) | Гипотеза | Задача / часть | Модель / recipe | Метрика | Статус | Отчёт |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-23T16:04:49+03:00 | HYP-001 | regression / inner | catboost/baseline | rmse=45.518715 | completed | [отчёт](2026-10-23/HYP-001-01-regression-inner-baseline-catboost/report.md) |
| 2026-10-23T16:04:57+03:00 | HYP-001 | regression / inner | catboost/correlation | rmse=45.320523 | completed | [отчёт](2026-10-23/HYP-001-02-regression-inner-correlation-catboost/report.md) |
| 2026-10-23T16:05:05+03:00 | HYP-002 | regression / inner | catboost/vif | rmse=45.000827 | completed | [отчёт](2026-10-23/HYP-002-01-regression-inner-vif-catboost/report.md) |
| 2026-10-23T16:05:10+03:00 | HYP-003 | regression / inner | catboost/scaled | rmse=45.000847 | completed | [отчёт](2026-10-23/HYP-003-01-regression-inner-scaled-catboost/report.md) |
| 2026-10-23T16:05:15+03:00 | HYP-004 | regression / inner | catboost/gaussian | rmse=45.122835 | completed | [отчёт](2026-10-23/HYP-004-01-regression-inner-gaussian-catboost/report.md) |
| 2026-10-23T16:05:21+03:00 | HYP-005 | regression / inner | catboost/engineered | rmse=45.263984 | completed | [отчёт](2026-10-23/HYP-005-01-regression-inner-engineered-catboost/report.md) |
| 2026-10-23T16:05:27+03:00 | HYP-001 | regression / inner | forest/baseline | rmse=46.163929 | completed | [отчёт](2026-10-23/HYP-001-03-regression-inner-baseline-forest/report.md) |
| 2026-10-23T16:05:50+03:00 | HYP-001 | regression / inner | forest/correlation | rmse=46.362483 | completed | [отчёт](2026-10-23/HYP-001-04-regression-inner-correlation-forest/report.md) |
| 2026-10-23T16:05:58+03:00 | HYP-001 | multiclass / inner | catboost/baseline | macro_f1=0.528302 | completed | [отчёт](2026-10-23/HYP-001-05-multiclass-inner-baseline-catboost/report.md) |
| 2026-10-23T16:06:08+03:00 | HYP-002 | regression / inner | forest/vif | rmse=46.685331 | completed | [отчёт](2026-10-23/HYP-002-02-regression-inner-vif-forest/report.md) |
| 2026-10-23T16:06:23+03:00 | HYP-003 | regression / inner | forest/scaled | rmse=46.679706 | completed | [отчёт](2026-10-23/HYP-003-02-regression-inner-scaled-forest/report.md) |
| 2026-10-23T16:06:37+03:00 | HYP-004 | regression / inner | forest/gaussian | rmse=46.711833 | completed | [отчёт](2026-10-23/HYP-004-02-regression-inner-gaussian-forest/report.md) |
| 2026-10-23T16:06:53+03:00 | HYP-005 | regression / inner | forest/engineered | rmse=46.946364 | completed | [отчёт](2026-10-23/HYP-005-02-regression-inner-engineered-forest/report.md) |
| 2026-10-23T16:07:08+03:00 | HYP-006 | regression / inner | catboost/vif (1) | rmse=44.558828 | completed | [отчёт](2026-10-23/HYP-006-01-regression-inner-vif-catboost-1/report.md) |
| 2026-10-23T16:07:11+03:00 | HYP-006 | regression / inner | catboost/vif (10) | rmse=45.603992 | completed | [отчёт](2026-10-23/HYP-006-02-regression-inner-vif-catboost-10/report.md) |
| 2026-10-23T16:07:15+03:00 | HYP-006 | regression / inner | catboost/vif (30) | rmse=46.330352 | completed | [отчёт](2026-10-23/HYP-006-03-regression-inner-vif-catboost-30/report.md) |
| 2026-10-23T16:07:19+03:00 | HYP-007 | regression / inner | l1/engineered (0.01) | rmse=611.145035 | completed | [отчёт](2026-10-23/HYP-007-01-regression-inner-engineered-l1-0.01/report.md) |
| 2026-10-23T16:07:20+03:00 | HYP-007 | regression / inner | l1/engineered (0.1) | rmse=517.428404 | completed | [отчёт](2026-10-23/HYP-007-02-regression-inner-engineered-l1-0.1/report.md) |
| 2026-10-23T16:07:21+03:00 | HYP-007 | regression / inner | l1/engineered (1) | rmse=54.197138 | completed | [отчёт](2026-10-23/HYP-007-03-regression-inner-engineered-l1-1/report.md) |
| 2026-10-23T16:07:21+03:00 | HYP-007 | regression / inner | l2/engineered (0.1) | rmse=878.177037 | completed | [отчёт](2026-10-23/HYP-007-04-regression-inner-engineered-l2-0.1/report.md) |
| 2026-10-23T16:07:21+03:00 | HYP-007 | regression / inner | l2/engineered (1) | rmse=884.672960 | completed | [отчёт](2026-10-23/HYP-007-05-regression-inner-engineered-l2-1/report.md) |
| 2026-10-23T16:07:21+03:00 | HYP-007 | regression / inner | l2/engineered (10) | rmse=719.024794 | completed | [отчёт](2026-10-23/HYP-007-06-regression-inner-engineered-l2-10/report.md) |
| 2026-10-23T16:07:22+03:00 | HYP-001 | regression / outer | catboost/baseline | rmse=45.935836 | completed | [отчёт](2026-10-23/HYP-001-06-regression-outer-baseline-catboost/report.md) |
| 2026-10-23T16:07:34+03:00 | HYP-005 | regression / outer | catboost/engineered | rmse=46.999742 | completed | [отчёт](2026-10-23/HYP-005-03-regression-outer-engineered-catboost/report.md) |
| 2026-10-23T16:07:45+03:00 | HYP-002 | regression / outer | catboost/vif | rmse=54.628428 | completed | [отчёт](2026-10-23/HYP-002-03-regression-outer-vif-catboost/report.md) |
| 2026-10-23T16:07:54+03:00 | HYP-001 | regression / outer | forest/baseline | rmse=48.565981 | completed | [отчёт](2026-10-23/HYP-001-07-regression-outer-baseline-forest/report.md) |
| 2026-10-23T16:08:36+03:00 | HYP-005 | regression / outer | forest/engineered | rmse=46.877846 | completed | [отчёт](2026-10-23/HYP-005-04-regression-outer-engineered-forest/report.md) |
| 2026-10-23T16:09:01+03:00 | HYP-006 | regression / outer | catboost/vif (1) | rmse=52.942394 | completed | [отчёт](2026-10-23/HYP-006-04-regression-outer-vif-catboost-1/report.md) |
| 2026-10-23T16:09:06+03:00 | HYP-005 | regression / полное обучение | catboost/vif (1) | Не измерялась | completed | [отчёт](2026-10-23/HYP-005-05-regression-full-refit/report.md) |
| 2026-10-23T16:09:06+03:00 | HYP-007 | regression / outer | l1/engineered (1) | rmse=58.664648 | completed | [отчёт](2026-10-23/HYP-007-07-regression-outer-engineered-l1-1/report.md) |
| 2026-10-23T16:09:06+03:00 | HYP-007 | regression / outer | l2/engineered (10) | rmse=65.452604 | completed | [отчёт](2026-10-23/HYP-007-08-regression-outer-engineered-l2-10/report.md) |
| 2026-10-23T16:09:12+03:00 | HYP-001 | multiclass / inner | catboost/correlation | macro_f1=0.513234 | completed | [отчёт](2026-10-23/HYP-001-08-multiclass-inner-correlation-catboost/report.md) |
| 2026-10-23T16:11:22+03:00 | HYP-002 | multiclass / inner | catboost/vif | macro_f1=0.513234 | completed | [отчёт](2026-10-23/HYP-002-04-multiclass-inner-vif-catboost/report.md) |
| 2026-10-23T16:13:27+03:00 | HYP-003 | multiclass / inner | catboost/scaled | macro_f1=0.510766 | completed | [отчёт](2026-10-23/HYP-003-03-multiclass-inner-scaled-catboost/report.md) |
| 2026-10-23T16:15:36+03:00 | HYP-004 | multiclass / inner | catboost/gaussian | macro_f1=0.512829 | completed | [отчёт](2026-10-23/HYP-004-03-multiclass-inner-gaussian-catboost/report.md) |
| 2026-10-23T16:17:28+03:00 | HYP-005 | multiclass / inner | catboost/engineered | macro_f1=0.512829 | completed | [отчёт](2026-10-23/HYP-005-06-multiclass-inner-engineered-catboost/report.md) |
| 2026-10-23T16:18:48+03:00 | HYP-001 | multiclass / inner | forest/baseline | macro_f1=0.602351 | completed | [отчёт](2026-10-23/HYP-001-09-multiclass-inner-baseline-forest/report.md) |
| 2026-10-23T16:19:33+03:00 | HYP-001 | multiclass / inner | forest/correlation | macro_f1=0.561459 | completed | [отчёт](2026-10-23/HYP-001-10-multiclass-inner-correlation-forest/report.md) |
| 2026-10-23T16:20:09+03:00 | HYP-002 | multiclass / inner | forest/vif | macro_f1=0.561459 | completed | [отчёт](2026-10-23/HYP-002-05-multiclass-inner-vif-forest/report.md) |
| 2026-10-23T16:20:43+03:00 | HYP-003 | multiclass / inner | forest/scaled | macro_f1=0.554821 | completed | [отчёт](2026-10-23/HYP-003-04-multiclass-inner-scaled-forest/report.md) |
| 2026-10-23T16:21:12+03:00 | HYP-004 | multiclass / inner | forest/gaussian | macro_f1=0.562953 | completed | [отчёт](2026-10-23/HYP-004-04-multiclass-inner-gaussian-forest/report.md) |
| 2026-10-23T16:21:55+03:00 | HYP-005 | multiclass / inner | forest/engineered | macro_f1=0.562953 | completed | [отчёт](2026-10-23/HYP-005-07-multiclass-inner-engineered-forest/report.md) |
| 2026-10-23T16:22:45+03:00 | HYP-006 | multiclass / inner | catboost/baseline (1) | macro_f1=0.541821 | completed | [отчёт](2026-10-23/HYP-006-05-multiclass-inner-baseline-catboost-1/report.md) |
| 2026-10-23T16:24:17+03:00 | HYP-008 | regression / inner | catboost/engineering_only | rmse=44.976675 | completed | [отчёт](2026-10-23/HYP-008-01-regression-inner-engineering_only-catboost/report.md) |
| 2026-10-23T16:24:26+03:00 | HYP-008 | regression / inner | forest/engineering_only | rmse=46.123522 | completed | [отчёт](2026-10-23/HYP-008-02-regression-inner-engineering_only-forest/report.md) |
| 2026-10-23T16:25:23+03:00 | HYP-006 | multiclass / inner | catboost/baseline (10) | macro_f1=0.509502 | completed | [отчёт](2026-10-23/HYP-006-06-multiclass-inner-baseline-catboost-10/report.md) |
| 2026-10-23T16:26:48+03:00 | HYP-008 | regression / outer | catboost/engineering_only | rmse=45.204915 | completed | [отчёт](2026-10-23/HYP-008-03-regression-outer-engineering_only-catboost/report.md) |
| 2026-10-23T16:27:03+03:00 | HYP-008 | regression / outer | forest/engineering_only | rmse=46.708293 | completed | [отчёт](2026-10-23/HYP-008-04-regression-outer-engineering_only-forest/report.md) |
| 2026-10-23T16:27:57+03:00 | HYP-005 | regression / полное обучение | catboost/engineering_only | Не измерялась | completed | [отчёт](2026-10-23/HYP-005-08-regression-full-refit/report.md) |
| 2026-10-23T16:28:18+03:00 | HYP-006 | multiclass / inner | catboost/baseline (30) | macro_f1=0.482148 | completed | [отчёт](2026-10-23/HYP-006-07-multiclass-inner-baseline-catboost-30/report.md) |
| 2026-10-23T16:29:09+03:00 | HYP-008 | multiclass / inner | catboost/engineering_only | macro_f1=0.526516 | completed | [отчёт](2026-10-23/HYP-008-05-multiclass-inner-engineering_only-catboost/report.md) |
| 2026-10-23T16:31:36+03:00 | HYP-007 | multiclass / inner | l1/engineered (0.01) | macro_f1=0.376382 | completed | [отчёт](2026-10-23/HYP-007-09-multiclass-inner-engineered-l1-0.01/report.md) |
| 2026-10-23T16:32:56+03:00 | HYP-008 | multiclass / inner | forest/engineering_only | macro_f1=0.594620 | completed | [отчёт](2026-10-23/HYP-008-06-multiclass-inner-engineering_only-forest/report.md) |
| 2026-10-23T16:35:11+03:00 | HYP-007 | multiclass / inner | l1/engineered (0.1) | macro_f1=0.375598 | completed | [отчёт](2026-10-23/HYP-007-10-multiclass-inner-engineered-l1-0.1/report.md) |
| 2026-10-23T16:39:28+03:00 | HYP-007 | multiclass / inner | l1/engineered (1) | macro_f1=0.377532 | completed | [отчёт](2026-10-23/HYP-007-11-multiclass-inner-engineered-l1-1/report.md) |
| 2026-10-23T16:43:51+03:00 | HYP-007 | multiclass / inner | l2/engineered (0.01) | macro_f1=0.376221 | completed | [отчёт](2026-10-23/HYP-007-12-multiclass-inner-engineered-l2-0.01/report.md) |
| 2026-10-23T16:46:35+03:00 | HYP-007 | multiclass / inner | l2/engineered (0.1) | macro_f1=0.377515 | completed | [отчёт](2026-10-23/HYP-007-13-multiclass-inner-engineered-l2-0.1/report.md) |
| 2026-10-23T16:49:34+03:00 | HYP-007 | multiclass / inner | l2/engineered (1) | macro_f1=0.377503 | completed | [отчёт](2026-10-23/HYP-007-14-multiclass-inner-engineered-l2-1/report.md) |
| 2026-10-23T16:52:33+03:00 | HYP-001 | multiclass / outer | catboost/baseline | macro_f1=0.523953 | completed | [отчёт](2026-10-23/HYP-001-11-multiclass-outer-baseline-catboost/report.md) |
| 2026-10-23T16:56:16+03:00 | HYP-005 | multiclass / outer | catboost/engineered | macro_f1=0.507527 | completed | [отчёт](2026-10-23/HYP-005-09-multiclass-outer-engineered-catboost/report.md) |
| 2026-10-23T16:56:45+03:00 | HYP-008 | multiclass / outer | catboost/engineering_only | macro_f1=0.523929 | completed | [отчёт](2026-10-23/HYP-008-07-multiclass-outer-engineering_only-catboost/report.md) |
| 2026-10-23T16:58:50+03:00 | HYP-001 | multiclass / outer | forest/baseline | macro_f1=0.572184 | completed | [отчёт](2026-10-23/HYP-001-12-multiclass-outer-baseline-forest/report.md) |
| 2026-10-23T17:00:25+03:00 | HYP-005 | multiclass / outer | forest/engineered | macro_f1=0.559498 | completed | [отчёт](2026-10-23/HYP-005-10-multiclass-outer-engineered-forest/report.md) |
| 2026-10-23T17:01:13+03:00 | HYP-008 | multiclass / outer | forest/engineering_only | macro_f1=0.577366 | completed | [отчёт](2026-10-23/HYP-008-08-multiclass-outer-engineering_only-forest/report.md) |
| 2026-10-23T17:01:30+03:00 | HYP-006 | multiclass / outer | catboost/baseline (1) | macro_f1=0.527088 | completed | [отчёт](2026-10-23/HYP-006-08-multiclass-outer-baseline-catboost-1/report.md) |
| 2026-10-23T17:02:59+03:00 | HYP-005 | multiclass / полное обучение | forest/baseline | Не измерялась | completed | [отчёт](2026-10-23/HYP-005-11-multiclass-full-refit/report.md) |
| 2026-10-23T17:05:21+03:00 | HYP-007 | multiclass / outer | l1/engineered (1) | macro_f1=0.371641 | completed | [отчёт](2026-10-23/HYP-007-15-multiclass-outer-engineered-l1-1/report.md) |
| 2026-10-23T17:11:00+03:00 | HYP-007 | multiclass / outer | l2/engineered (0.1) | macro_f1=0.371543 | completed | [отчёт](2026-10-23/HYP-007-16-multiclass-outer-engineered-l2-0.1/report.md) |
| 2026-10-23T17:15:00+03:00 | HYP-005 | multiclass / полное обучение | forest/baseline | Не измерялась | completed | [отчёт](2026-10-23/HYP-005-12-multiclass-full-refit/report.md) |

## Новые семейства и доставка

[Единый журнал гипотез](../HYPOTHESES.md) читать перед КАЖДЫМ экспериментом; [новый итоговый выбор](2026-10-23/NEW_EXPERIMENT_REPORT.md).

| Run | Метрика | Отчёт |
| --- | --- | --- |
| HYP-009-01-regression-lightgbm | rmse=60.492688256313926 | [отчёт](2026-10-23/HYP-009-01-regression-lightgbm/report.md) |
| HYP-010-01-regression-extra_trees | rmse=43.56756773832011 | [отчёт](2026-10-23/HYP-010-01-regression-extra_trees/report.md) |
| HYP-009-02-multiclass-lightgbm | macro_f1=0.5838559593680208 | [отчёт](2026-10-23/HYP-009-02-multiclass-lightgbm/report.md) |
| HYP-010-02-multiclass-extra_trees | macro_f1=0.5804952756940901 | [отчёт](2026-10-23/HYP-010-02-multiclass-extra_trees/report.md) |

- SPEC-004 HYP-011/regression: completed, rmse=43.510161481164815; [отчёт](2026-10-23/HYP-011-01-regression-none/report.md).

- SPEC-004 HYP-011/multiclass: completed, macro_f1=0.5914513709191194; [отчёт](2026-10-23/HYP-011-02-multiclass-none/report.md).

- SPEC-004 HYP-012/regression: completed, rmse=42.19670002719412; [отчёт](2026-10-23/HYP-012-01-regression-log1p_target/report.md).

- SPEC-004 HYP-013/regression: completed, rmse=47.18831919578777; [отчёт](2026-10-23/HYP-013-01-regression-none/report.md).

- SPEC-004 HYP-014/regression: completed, rmse=178.08498786616641; [отчёт](2026-10-23/HYP-014-01-regression-linear_scaled/report.md).

- SPEC-004 HYP-015/regression: completed, rmse=44.402896602005654; [отчёт](2026-10-23/HYP-015-01-regression-target_encode/report.md).

- SPEC-004 HYP-016/multiclass: completed, macro_f1=0.5807635957092588; [отчёт](2026-10-23/HYP-016-01-multiclass-categorical_cross/report.md).

- SPEC-004 HYP-017/multiclass: completed, macro_f1=0.5909684499124109; [отчёт](2026-10-23/HYP-017-01-multiclass-frequency/report.md).

- SPEC-004 HYP-018/multiclass: completed, macro_f1=0.578639858804918; [отчёт](2026-10-23/HYP-018-01-multiclass-none/report.md).

- SPEC-004 HYP-019/multiclass: completed, macro_f1=0.5901110420481382; [отчёт](2026-10-23/HYP-019-01-multiclass-hierarchical/report.md).

- SPEC-004 HYP-020/multiclass: completed, macro_f1=0.5829349072076985; [отчёт](2026-10-23/HYP-020-01-multiclass-inverse_group_weights/report.md).

- SPEC-006 HYP-022/regression: completed, rmse=42.203267904780695; Не подтверждена по заданному критерию; [отчёт](2026-10-23/HYP-022-01-regression-spec006/report.md).

- SPEC-006 HYP-022/multiclass: completed, macro_f1=0.5615525355321143; Не подтверждена по заданному критерию; [отчёт](2026-10-23/HYP-022-02-multiclass-spec006/report.md).

- SPEC-006 HYP-023/regression: completed, rmse=43.19210947337134; Не подтверждена по заданному критерию; [отчёт](2026-10-23/HYP-023-01-regression-spec006/report.md).

- SPEC-006 HYP-024/regression: completed, rmse=42.18354628369281; Подтверждена в проверенных условиях; [отчёт](2026-10-23/HYP-024-01-regression-spec006/report.md).

- SPEC-006 HYP-025/multiclass: completed, macro_f1=0.5771726314994396; Не подтверждена по заданному критерию; [отчёт](2026-10-23/HYP-025-01-multiclass-spec006/report.md).

- SPEC-006 HYP-026/multiclass: completed, macro_f1=0.5793987048589155; Не подтверждена по заданному критерию; [отчёт](2026-10-23/HYP-026-01-multiclass-spec006/report.md).

- SPEC-006 HYP-027/multiclass: completed, macro_f1=0.5857385909681803; Не подтверждена по заданному критерию; [отчёт](2026-10-23/HYP-027-01-multiclass-spec006/report.md).

- SPEC-006 HYP-028/regression: failed, None=None; Недостаточно данных; [отчёт](2026-10-23/HYP-028-01-regression-spec006/report.md).

- SPEC-006 HYP-028/multiclass: failed, None=None; Недостаточно данных; [отчёт](2026-10-23/HYP-028-02-multiclass-spec006/report.md).

- SPEC-006 HYP-029/regression: completed, rmse=42.08610226703874; Подтверждена в проверенных условиях; [отчёт](2026-10-23/HYP-029-01-regression-spec006/report.md).

- SPEC-006 HYP-030/regression: completed, rmse=41.993466003366024; Подтверждена в проверенных условиях; [отчёт](2026-10-23/HYP-030-01-regression-spec006/report.md).

- SPEC-006 HYP-031/regression: completed, rmse=42.00460393727476; Подтверждена в проверенных условиях; [отчёт](2026-10-23/HYP-031-01-regression-spec006/report.md).

- SPEC-006 HYP-032/regression: completed, rmse=43.810373477573165; Не подтверждена по заданному критерию; [отчёт](2026-10-23/HYP-032-01-regression-spec006/report.md).

- SPEC-006 HYP-033/regression: completed, rmse=42.974266298635236; Не подтверждена по заданному критерию; [отчёт](2026-10-23/HYP-033-01-regression-spec006/report.md).

- SPEC-006 HYP-034/regression: completed, rmse=44.42393495521875; Не подтверждена по заданному критерию; [отчёт](2026-10-23/HYP-034-01-regression-spec006/report.md).

- SPEC-006 HYP-035/regression: completed, rmse=41.480963027907194; Подтверждена в проверенных условиях; [отчёт](2026-10-23/HYP-035-01-regression-spec006/report.md).

- SPEC-006 HYP-036/regression: completed, rmse=42.25509131431627; Не подтверждена по заданному критерию; [отчёт](2026-10-23/HYP-036-01-regression-spec006/report.md).

- SPEC-006 HYP-037/multiclass: completed, macro_f1=0.6286047899141216; Подтверждена в проверенных условиях; [отчёт](2026-10-23/HYP-037-01-multiclass-spec006/report.md).

- SPEC-006 HYP-038/multiclass: completed, macro_f1=0.595178021245499; Подтверждена в проверенных условиях; [отчёт](2026-10-23/HYP-038-01-multiclass-spec006/report.md).

- SPEC-006 HYP-039/multiclass: completed, macro_f1=0.5795834428470248; Не подтверждена по заданному критерию; [отчёт](2026-10-23/HYP-039-01-multiclass-spec006/report.md).

- SPEC-006 HYP-040/multiclass: completed, macro_f1=0.5922402460598002; Подтверждена в проверенных условиях; [отчёт](2026-10-23/HYP-040-01-multiclass-spec006/report.md).

- SPEC-006 HYP-041/regression: completed, rmse=42.19670002719412; Не подтверждена по заданному критерию; [отчёт](2026-10-23/HYP-041-01-regression-spec006/report.md).

- SPEC-006 HYP-041/multiclass: completed, macro_f1=0.5914513709191194; Не подтверждена по заданному критерию; [отчёт](2026-10-23/HYP-041-02-multiclass-spec006/report.md).

- SPEC-006 HYP-028/regression, техническое продолжение безповтораinner: completed, rmse=42.245303736669484; Не подтверждена по заданному критерию; [отчёт](../results/2026-10-23/HYP-028-03-regression-guard-continuation/report.md). Повтор не запускать.

- SPEC-006 HYP-028/multiclass, техническое продолжение безповтораinner: completed, macro_f1=0.5676156935900548; Не подтверждена по заданному критерию; [отчёт](../results/2026-10-23/HYP-028-04-multiclass-guard-continuation/report.md). Повтор не запускать.

| HYP-042 | regression | completed, rmse=41.5048413147343; Не подтверждена по заданному критерию | [report](../results/2026-10-23/HYP-042-01-regression-combinations/report.md) |

| HYP-043 | regression | completed, rmse=40.972997661684175; Подтверждена в проверенных условиях | [report](../results/2026-10-23/HYP-043-01-regression-combinations/report.md) |

| HYP-044 | regression | completed, rmse=41.94416116050523; Не подтверждена по заданному критерию | [report](../results/2026-10-23/HYP-044-01-regression-combinations/report.md) |

| HYP-045 | regression | completed, rmse=41.406800898690484; Подтверждена в проверенных условиях | [report](../results/2026-10-23/HYP-045-01-regression-combinations/report.md) |

| HYP-046 | multiclass | completed, macro_f1=0.6154359833457845; Не подтверждена по заданному критерию | [report](../results/2026-10-23/HYP-046-01-multiclass-combinations/report.md) |

| HYP-047 | multiclass | completed, macro_f1=0.6336560111990485; Подтверждена в проверенных условиях | [report](../results/2026-10-23/HYP-047-01-multiclass-combinations/report.md) |

| HYP-048 | multiclass | completed, macro_f1=0.6495311620074445; Подтверждена в проверенных условиях | [report](../results/2026-10-23/HYP-048-01-multiclass-combinations/report.md) |

| HYP-069 | regression | completed, rmse=39.39219382766771; Подтверждена в проверенных условиях | [report](../results/2026-10-24/HYP-069-01-regression-alternatives/report.md) |

| HYP-049 | multiclass | completed, macro_f1=0.6149071035851954; Не подтверждена по заданному критерию | [report](../results/2026-10-24/HYP-049-01-multiclass-alternatives/report.md) |

| HYP-051 | multiclass | completed, macro_f1=0.5632041224667611; Не подтверждена по заданному критерию | [report](../results/2026-10-24/HYP-051-01-multiclass-alternatives/report.md) |

| HYP-053 | multiclass | failed, None=None; Недостаточно данных | [report](../results/2026-10-24/HYP-053-01-multiclass-alternatives/report.md) |

| HYP-054 | regression | completed, rmse=44.388898407039605; Не подтверждена по заданному критерию | [report](../results/2026-10-24/HYP-054-01-regression-alternatives/report.md) |

| HYP-055 | regression | completed, rmse=47.44104483605545; Не подтверждена по заданному критерию | [report](../results/2026-10-24/HYP-055-01-regression-alternatives/report.md) |

| HYP-056 | regression | completed, rmse=43.063770695804266; Не подтверждена по заданному критерию | [report](../results/2026-10-24/HYP-056-01-regression-alternatives/report.md) |

| HYP-058 | regression | completed, rmse=42.36424466952146; Не подтверждена по заданному критерию | [report](../results/2026-10-24/HYP-058-01-regression-alternatives/report.md) |

| HYP-060 | regression | completed, rmse=42.46179807398836; Не подтверждена по заданному критерию | [report](../results/2026-10-24/HYP-060-01-regression-alternatives/report.md) |

| HYP-060 | multiclass | completed, macro_f1=0.567987406329944; Не подтверждена по заданному критерию | [report](../results/2026-10-24/HYP-060-02-multiclass-alternatives/report.md) |

| HYP-065 | schema diagnosis6/8→fixed8/8 | model fits0 | [proof](2026-10-24/HYP-065-02-schema-codec-fix/verification.json) |
| HYP-068 | platformprovenance incomplete, score1376.762oldimprovedZIP | model fits0 | [report](2026-10-24/HYP-068-01-platform-provenance/report.md) |
