# Полный запрошенный вариант предобработки

Включены imputation, correlation, VIF<=10, StandardScaler, train-fit Yeo–Johnson и feature engineering.
Статистики обучены только на полном исходном train; test только преобразован.
Этот экспорт предназначен для дальнейшего полного обучения и inference, не для оценки сохранённых validation.
Он сохранён независимо от того, улучшил ли полный pipeline метрику и выбран ли он для финальной модели.

[Массивы](preprocessed_data.npz), [препроцессор](preprocessor.joblib), [аудит](preprocessing_audit.json), [manifest](manifest.json).
NPZ содержит X_train, X_test, y_train, train_id, test_id, feature_names; классы y_train хранятся строками/объектами.
Для безопасного чтения числовых массивов используйте numpy.load(..., allow_pickle=False); y_train при необходимости берите из исходного CSV.
