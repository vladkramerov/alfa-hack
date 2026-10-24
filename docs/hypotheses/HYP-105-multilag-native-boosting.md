# HYP-105: exact multi-lag history и native missing LightGBM

Дата2026-10-24; SPEC013. До fit: зарегистрирована, не оценивалась.
Идея из hilhool-agent@17bcb1, code/agent/entity.py add_lags и
code/agent/core.py prepare; самостоятельная реализация без заимствования кода.

H76 хранит последнее доступное наблюдение и все weather-delta в CatBoost;
SergeyH72 использует EWmean history residualLGB, H74 — ET по EWstate.
Здесь exact1/2/3 календарных периода позволяют отличить последовательную
историю от старого donor и изменение режима; оригинальный target обучается
по всем fit-строкам, включая cold, nativeNaN без насильственной медианы.

Ключ State_Factor/building_class/facility_type/floor_area/year_built,
безID/ELEVATION. Пропущенные компоненты иyear_built0 не совпадают;
любой key/year collision исключает весьпрофиль изистории, строки не удаляются.
Каждый featurehistory использует только fitrecord.year<query.year.

Features: rawX безYear_Factor + lag1/2/3 target, previousavailable target,
mean/std/min/max/count/gap и last-minus-previous trend;3 numericchangingfields
(≥100 adjacentvalidpairs, changefrequency[.01,.99], largestabsSpearman(y),
lexicalties) с EXACTlag1value/current−lag1value. Keys/year/ID excluded.
Выбор полей и категорий только fit. Missing numeric→NaN, unseen category→NaN.

FixedLGB1200/.03/leaves63/minchild20/feature_fraction.8/bagging.8freq1/
lambdaL2=5/seed42/threads4/objectiveoriginalRMSE; noearlystop/noqueryadaptation.
Два preregisteredarms pure и.75new+.25cachedSergeyH73. Один learnerfit/stage;
arm выбирается только inner, outer лишь если лучший <30.636363541581975.
Сначала Year<4→4, потом onefrozenarmYear<5→5. Общий900сек/H105.

Критерий: RMSE нижеSergey31.046636/34.520269 и исследовательскогоH91
30.636364/34.213828. Цельминимум5%надSergey — отдельно ≤32.79425522185614
наouter; меньшийwinчестноотмечать. Fullpreprocessing/outlier5% не возвращается.
Все результаты, отрицательные исходы и SHA сохраняются; безhiddenclaims.

HYP-105/inner: {"status": "completed", "hypothesis": "HYP-105", "stage": "inner", "metric": "rmse", "score": 32.01672432373479, "arm_scores": {"pure": 34.56623580026114, "blend75": 32.01672432373479}, "selected_arm": "blend75", "control_score": 31.04663635241995, "selected_on": "inner_only", "learner_fits": 1, "old_model_refits": 0, "fit_rows": 22054, "valid_rows": 12946, "eligible_for_outer": false, "seconds": 8.094000000011874, "fingerprint": "c0523370162349ee4ad3a8f474ce60306f6f816758b3a35777060ad20b0228de"}; [report](../../results/2026-10-24/HYP-105-01-inner-external-multilag/report.md). Не повторять.
