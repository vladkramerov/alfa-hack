# Sprint adapter verification: PASS

Команда: `.venv-ml/Scripts/python.exe -X utf8 -m experiments.verify_sprint_adapter`.

Проверены исходные SHA данных, source SHA и exact splits; все fit защищены исключением. H78 воспроизводится из трёх сохранённых ветвей/H48 meta и frozen H78 theta/gain/offset. H70 воспроизводится из сохранённого linker alpha.95 и H69; H69 проверен через cached memory и существующий H43 model inference. H91 проверяет frozen inner weights, неизменные defaults H70/H83/H85, их сохранённые модели/сырые predictions и pool dot-product; дополнительно H70/H83 в памяти получают один общий reconstructed H69 predictor. Численные итоги и ошибки: [verification.json](verification.json).

Это replay уже измеренного outer-графа; детерминизм и качество будущего full-data refit не установлены. Новых гипотез/модельных fit/научных метрик нет.
