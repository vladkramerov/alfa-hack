# HYP-110: полный TabM с весами редких классов

Дата24.10.2026, статусЗапланирована. [SPEC016](../specs/SPEC-016-torch-tabm-delivery.md).
Та же full adapters/learned PWL16d numeric architecture H109; train-only class weights1/sqrt(count), mean sample weight1; CE, не teacher-residual softF1MLP H72/H94 и не H108mini. k8/width96/blocks2/dropout.1/60epochs/batch256/AdamW.001/wd.0001/CPU4/≤120sec fit. Без удаления строк/классов, median/missing/OHE64 fit-only. SHA171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0, grouped H78 inner/outer.
Контроль — реальный предыдущийZIP probability_stacking с fresh train-only OOF, а не cached H78 decision wrapper. Прямое поручение пользователя разрешает однократное сопоставимое измерение контроля.
Критерий доfit: inner gridblend0/.05/.1/.2/.4/.7/1 строго улучшает Macro-F1, новая доля>0; выбранный вес freeze доouter. Outer строго больше ZIP control. При проигрыше сохранить прежнюю class часть. ≤900sec/hyp. Исход: Не оценивалась.
