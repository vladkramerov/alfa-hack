# Полное обучение и предсказания

Конфигурация выбрана и заморожена по inner до оценки outer. Это техническое обучение на всём train, не новая проверка гипотезы.

[Метаданные](metrics.json), [submission](submission.csv), [pipeline](model.joblib), [преобразованные массивы](preprocessed_data.npz), [аудит](preprocessing_audit.json).

Дополнение: это дополнительный разведочный кандидат HYP-008; основной frozen winner сохранён. Выбор сделан по повторно используемому inner до новых outer-оценок; качество hidden test неизвестно. [Происхождение](supplemental-provenance.json).
