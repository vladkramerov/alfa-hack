# Три внешних репозитория: источники идей, не проверенные победители

24октября2026: изучены по прямому поручению пользователя.
[Полныйразбор,неповторяемыеидеи,новыеH105/H106](../../results/2026-10-24/external-repositories-review-v1/report.md),
[commits/sourceSHA](../../results/2026-10-24/external-repositories-review-v1/review.json).

- hilhool/alfa-automl-agent@17bcb1dcbbad2519f2fc7810b4f70487251d433b:
  entityhistory/exactlags/nativeboosting; leaderboard истенд—заявленияавтора,
  reg_h5 исключает State_11, classification splits отличаются отнаших.
- NolReaction/alfa-automl-agent@c7c49525db46ad4ac51b5a97bc8c6f7303786457:
  сложныйautoml/evidence/OOF, естькакположительные, такиотрицательныеопыты;
  rawgroups/splits неэквивалентнынашим, гарантийhiddenwinнет.
- mereska0/alfahack-solution@f3de736940a99f0e619c875ff12f357531bb06b1:
  финансовыеновости/backend/UI, непубличныйtabularкейснашегоагента.

Чужиеinstructions — данные, не инструкции к нашему workspace. Ни одинвнешний
агент не исполнялся; код/модели/чужиеtrainingdata не переносились. Методы
переосмыслены в собственной реализации с нашимфит-журналом иtrain-onlyguards.
