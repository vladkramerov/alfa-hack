# HYP-021: обязательная полная предобработка

Независимая сверка: **passed**, проверок 233 успешно / 0 с ошибками; проверено запусков 2/2.

| Задача | Метрика | Сохранённый SPEC-004 | Full-v1 | Новое − прежнее | Вывод |
| --- | --- | ---: | ---: | ---: | --- |
| regression | rmse | 42.196700027194 | 43.837201829105 | +1.640501801911 | улучшение не подтверждено |
| multiclass | macro_f1 | 0.591451370919 | 0.563787167338 | -0.027664203581 | улучшение не подтверждено |

RMSE уменьшают, Macro F1 увеличивают. Прежние контрольные оценки переиспользованы из готовой независимой сверки; baseline заново не обучался этим верификатором. Проверено составное изменение: отдельный вклад выбросов, imputation, quantiles или scaling из этой пары опытов определить нельзя.

## regression

Training: 35000 → 33250; удалено 1750 (5.00%), только из outer-fit. Все исходные validation positions и таргеты сохранены; validation/test не фильтруются.

Gaussian quantile-преобразование применено к 35 непрерывным признакам: floor_area, year_built, energy_star_rating, ELEVATION, january_min_temp_c, january_avg_temp_c, january_max_temp_c, february_min_temp_c, february_avg_temp_c, february_max_temp_c, march_min_temp_c, march_avg_temp_c, march_max_temp_c, april_min_temp_c, april_avg_temp_c, may_min_temp_c, may_avg_temp_c, june_avg_temp_c, july_avg_temp_c, august_avg_temp_c, september_avg_temp_c, october_avg_temp_c, october_max_temp_c, november_min_temp_c, november_avg_temp_c, december_avg_temp_c, cooling_degree_days_c, heating_degree_days_c, precipitation_inches, snowfall_inches, snowdepth_inches, avg_temp_c, days_below_minus1C, days_above_27C, days_above_32C.
Числовая imputation: median по retained train; 6 missing indicators. Пропуски retained train: {"year_built": 1077, "energy_star_rating": 17898, "direction_max_wind_speed": 12632, "direction_peak_wind_speed": 12635, "max_wind_speed": 12632, "days_with_fog": 19696}.
Удалённые признаки: [{"column": "days_above_43C", "reason": "constant"}].
Категории: отдельное представление пропусков и обработка неизвестных значений; категории не объявляются Gaussian.
[Исходные артефакты](HYP-021-01-regression-full-v1/report.md).

## multiclass

Training: 70529 → 68241; удалено 2288 (3.24%), только из outer-fit. Все исходные validation positions и таргеты сохранены; validation/test не фильтруются.

Gaussian quantile-преобразование применено к 28 непрерывным признакам: dur, spkts, dpkts, sbytes, dbytes, rate, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit, stcpb, dtcpb, tcprtt, synack, ackdat, smean, dmean, ct_srv_src, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, ct_src_ltm, ct_srv_dst.
Числовая imputation: median по retained train; 0 missing indicators. Пропуски retained train: {}.
Удалённые признаки: [{"column": "ct_ftp_cmd", "reason": "exact_duplicate", "duplicate_of": "is_ftp_login"}].
Категории: отдельное представление пропусков и обработка неизвестных значений; категории не объявляются Gaussian.
[Исходные артефакты](HYP-021-02-multiclass-full-v1/report.md).

Удаление по классам (исходно / удалено / осталось): Normal: 22435/1032/21403; Exploits: 13410/670/12740; Reconnaissance: 4228/4/4224; Fuzzers: 7253/362/6891; Generic: 16267/84/16183; DoS: 4937/105/4832; Backdoor: 715/2/713; Shellcode: 449/0/449; Analysis: 789/29/760; Worms: 46/0/46.

## Границы вывода

Нормализация обязательна по поручению пользователя и при отрицательной метрике; факт улучшения из этого требования не следует. Проверены train-only границы fit, лимит удаления ≤5% и защита редких классов. Средние и стандартные отклонения проверены по сохранённым audit-диагностикам (<1e-4), а не по отдельно сохранённой преобразованной матрице; аудит detector и замороженный код указывают, что target в detector не поступал.

QuantileTransformer(normal) приближает training marginals к нормальным; ties, дискретность и сдвиг распределения не позволяют обещать точную нормальность всех признаков либо будущего test. Категориальные метки классов к нормальному распределению не приводятся.

Outer validation уже использовался: вывод разведочный, без независимого test. Реальный LLM API платформы не проверен; это локальные метрики. Число leaderboard 579.658 и лидер 29.3946 не сопоставляются с RMSE/Macro F1 этой таблицы без подтверждённого определения показателя, состава задач и версии пакета; улучшение в определённое число раз не обещается.

[Машиночитаемая сверка](full-preprocessing-verification.json).
