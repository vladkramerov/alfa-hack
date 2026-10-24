# HYP-108: TabM-mini и ансамбль с сохранённым лидером

Дата: 2026-10-24. Статус: Запланирована. Спецификация: [SPEC015](../specs/SPEC-015-tabm-trial.md).

Новый механизм: input-adapted parameter-sharing ensemble c независимыми головами; не teacher-residual MLP H72/H94, не FM H62, не повтор старых бустингов или их ансамблей. Самостоятельная NumPy-реализация архитектуры TabM-mini, без числового обещания соответствовать официальному PyTorch пакету.

Данные: regression SHA096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775; classification SHA171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0. Splits и контролируемые прогнозы: H107/H70 regression, H78 classification. Точные SHA артефактов и исходников будут зафиксированы evidence.json ДО каждого запуска.

Config: k8,hidden64,depth2,60epochs,batch256,AdamW.003,wd.0001,seed42; numeric median+missing+quantile-normal, categoricalOHE≤64/column; original standardized regression-y MSE, classCE. Нет row filtering, teacher fitting, ручных classweights или feature-selection по validation. ≤240сек/fit, ≤900сек/H108.

Критерий ДО запуска: inner selected pure/blend строго лучше cached H107RMSE30.59160391991867 или cached H78F1.6288481661155734, вес новой модели>0. Для blend регрессия analytical[0,1], классификация fixed grid0/.1/.25/.5/.75/1. Outer только прошедшей части; успех outer — строго лучше соответствующего cached control на идентичных строках. Regression лучше при снижении RMSE, classification при росте Macro-F1; остальные части поставки остаются текущими. Outer многократно наблюдался, вывод exploratory, не hidden gain.

Запуски: пока нет. Вывод: Не оценивалась.
