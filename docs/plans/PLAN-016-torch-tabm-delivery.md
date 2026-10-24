# PyTorch TabM comparison and delivery plan

Native execution; [SPEC016](../specs/SPEC-016-torch-tabm-delivery.md). User explicitly authorized experiments and publishing improvements, routine implementation choices delegated.

- [ ] Зафиксировать SHA прошлой поставки и однократно получить реальные ZIP inner/outer controls; отдельные reports и component fit journals.
- [ ] RED/GREEN tests: Piecewise bins fit-only, full adapter gradients/diversity, reg/class learning, label/OOV/schema/pickle, frozen blend and grouped/temporal splits.
- [ ] Реализовать automl_torch_tabm.py с полным efficient ensemble, обучаемыми numeric embeddings и CPU bounded training.
- [ ] Зарегистрировать H109/H110 ДО fit, snapshot source/config/data/splits; прочитать полный HYPOTHESES перед каждым обучением.
- [ ] Inner gating, frozen weight, outer only eligible; outcomes immediately append. Ни H108, ни старые конфигурации не повторять.
- [ ] При успехе создать новый flat delivery из предыдущего ZIP, добавить runtime train-only blend/factory/LLM schema/portfolio/history identity.
- [ ] Проверить exact runtime graph с reuse идентичных control/neural fitted states; technical full refits для clean ZIP без повторных quality fits.
- [ ] Tests/review/independent metric replay/root imports/CLI/requirements/submission checks, SHA and report.
- [ ] Опубликовать только новую поставку и связанные доказательства в ivan_branch, commit title с относительными выигрышами; проверить remote commit/ZIP SHA.

Review focus: rare labels absent from fold, raw-X duplicate leakage, future targets, mismatch between cached and delivered graph, timeout fallback, deserialization and new task schema, accidental unrelated WIP publication.
