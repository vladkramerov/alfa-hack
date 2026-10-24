# HYP-109: полный TabM + обучаемые кусочно-линейные embeddings

Дата24.10.2026, статусЗапланирована. [SPEC016](../specs/SPEC-016-torch-tabm-delivery.md).
Полный adapters-on-every-layer ensemble, numeric8quantilebins+learned16dprojection, а не первый input adapter H108. k8/hidden96/blocks2/dropout.1/60epochs/batch256/AdamW.001/wd.0001/CPU4/≤120sec fit; y original standardizedMSE. Median/missing/OHE64 fit-only, без row deletion. Данные SHA096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775; splitH70inner/outer.
Контроль — реальный runtime предыдущего ZIP (SHA в SPEC016), его однократное сравнение прямо поручено пользователем. Не cached H107, чья runtime quality не измерялась.
Критерий доfit: inner analytical convex blend с control строго снижает RMSE и новый вес>0; frozen доouter. Outer строго меньше control на тех же строках. При проигрыше — без новой reg поставки. ≤900sec/hyp. Никаких старых refits без отдельной роли разрешённого ZIP control; H108 не повторять.
Исход: Не оценивалась.
