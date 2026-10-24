# HYP-106: найти ключ истории по training data вместо фиксированного профиля

2026-10-24, SPEC013, зарегистрирована доfit; не оценивалась.
Самостоятельная реализация идеи hilhool/entity.find_key и fit-only discovery
из NolReaction; внешний код не исполняется и не копируется.

Новый механизм относительноSergeyH70/H73 иIvanH69/H70 — ключ не задан5/6
конкретными полями. Greedy≤6fields, cardinality2..0.9N, maximizecoverage/
meanrows_per_key_period; псевдоquery только последний период ВНУТРИfit,
историяпсевдоquery только болееранниеfitX. Улучшение шага>1%, финальный
coverage≥.2, среднееwithin-key std(y)/std(y)≤.5. Runtimequery не участвует
в подборе ключа/статистик. ID/target/time не key, missingkeyне совпадает.

Формулаhistorymean strictlypast сhalf-life1period иконсервативнойweight
min(.9,n/(n+1))*exp(-.25max(0,gap-1)). Повторkey/year усредняется как
неопределенныйпрофиль; совпадение не доказывает идентичностьфизическогообъекта.
Fallback cachedSergeyH73, oldlearnerrefits0, keyparameterfit1/stage.

Arms pureoverlay и.75overlay+.25Sergey заранее; innerYear<4→4 first,
onefrozenarm outerYear<5→5 onlyifinner<30.636363541581975. Goal5%отSergey
наouter≤32.79425522185614 отдельно отлюбого lesserwin. Max900secH106.
Новыйseed/переименование не новизна; исход и veto сохраняются вresults.

HYP-106/inner: {"status": "completed", "hypothesis": "HYP-106", "stage": "inner", "metric": "rmse", "score": 30.963253972816442, "arm_scores": {"pure": 31.15263063723188, "blend75": 30.963253972816442}, "selected_arm": "blend75", "control_score": 31.04663635241995, "selected_on": "inner_only", "learner_fits": 0, "parameter_fits": 1, "old_model_refits": 0, "fit_rows": 22054, "valid_rows": 12946, "eligible_for_outer": false, "seconds": 11.90600000001723, "fingerprint": "58cced6f80dc02bd615b7ca46705d6be831d11aa1377f735e38c63356c19de61"}; [report](../../results/2026-10-24/HYP-106-01-inner-external-key/report.md). Не повторять.
