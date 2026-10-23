# Результаты проверок

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
