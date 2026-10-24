# HYP-107: Сергей annual dynamics + наши rating transport / uncertainty

Статус: запланирована до parameter fit. SPEC014/PLAN014.

Итог: завершена; новая cached-композиция прошла оба гейта. [Inner](../../results/2026-10-24/HYP-107-01-inner-sergey-fusion/report.md):30.59160391991867, [outer](../../results/2026-10-24/HYP-107-02-outer-sergey-fusion/report.md):34.142888130110755. Подтверждена в проверенных условиях cached cross-OS; runtime-quality/hidden gain не доказаны. Не повторять.

Утверждение: ошибки Sergey H73 и Ivan H70/H83/H85 комплементарны; convex simplex pool улучшит лучшую отдельную композицию.
Новизна: Sergey H73 не входил в прежние H79/H91/H102. Это не повтор компонентов и не новый seed; learner fits0, один parameter fit.
Конфигурация: компоненты в порядке SergeyH73,IvanH70,IvanH83,IvanH85; MSE simplex SLSQP maxiter200 ftol1e-12, начальные равные веса. Без pruning, grid search, внешних данных, intercept и отрицательных весов.
Данные: regression train SHA096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775. Inner Year<4→4, outer Year<5→5; идентичные positions/splits, y atol1e-10 для Mac/Windows CSV. Все исходные predictions read-only.
Гейт inner: RMSE<30.636363541581975 и <31.04663635241995; ≥2веса>1e-8. Веса фиксируются прежде outer load. Успех outer: RMSE<34.21382789041039 и <34.520268654585415. Никакого дополнительного подбора после outer.
Бюджет:900sec суммарно,4CPU максимум; старые learnerrefits0. Отрицательный исход сохраняется и не повторяется.
Ограничение: адаптивно использованный public outer, cross-OS компоненты; улучшение платформы и перенос на hidden данные не доказаны. Классификация не затронута.
