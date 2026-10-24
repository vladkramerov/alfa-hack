# HYP-079: measured new temporal transport + supervised transition combination

SPEC011,2026-10-24. Registered before evaluating blend weights. Combines NEW H70(alpha.95) and H76 direct-y supervisedpanel predictions; unlike H44 generic booster convexblend, combines transport with a learned temporal conditional transition. Both positive inner vs savedH69, but H76 alone loses toH70. Whether their errors complement is unknown.

Inner uses saved predictions only (same Year<4→4). Fit convex weight w=clip(sum((p70-p76)*(y-p76))/sum((p70-p76)^2),0,1) analytically before any outer H76 score. Select combination ONLY if w strictlyinside(0,1) and RMSE strictlybeatsH70inner30.66325537997; otherwise negative, no outerfit. Freeze w+sourceSHA in separate report before any outerfit. Outer requires one previously unseenH76fit Year<5→5; reuse H70outer predictions and H69control, no repeat ofknownlearners. No inference-time labels, taskIDs or hiddenlabels; productionfixedw. Timecap900sec includes source componentnewfit.

Success outer RMSE<34.318489927675174 and usergoal≤31.513755 separately. Outer alreadyadaptivelyviewed; no independent generalization claim. Everyresult including negative saved; repeat forbidden.

2026-10-24 outcome: [inner report](../../results/2026-10-24/HYP-079-01-inner-combination-sprint/report.md). Optimizer выбрал H70=1,H76=0; incremental улучшения нет; outer не запускался, повтор запрещен.
