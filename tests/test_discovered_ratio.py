import numpy as np
import pandas as pd
from scipy.special import ndtri
from automl_ratio import DiscoveredRatio


def test_discovered_ratio_is_strictly_past_and_batch_invariant():
    rng = np.random.RandomState(91)
    n = 500
    frame = pd.DataFrame({'anchor': np.tile(np.arange(n), 3),
                          'clock': np.repeat([1, 2, 3], n),
                          'group': np.tile(np.where(np.arange(n) % 2, 'a', 'b'), 3),
                          'quality': rng.randint(1, 101, n*3).astype(float)})
    target = 20 * np.exp(.3 * ndtri(np.clip((frame.quality.to_numpy()+.5)/100, .005, .995)))
    frame.loc[[11, 12], 'quality'] = np.nan
    model = DiscoveredRatio('clock').fit(frame, target)
    assert model.driver_ == 'quality'
    assert model.audit_['percentile_support']
    assert np.isnan(model.train_values_[11:13]).all()
    # No same-time donor and no insertion of query rows into memory.
    first = frame.iloc[:n]
    np.testing.assert_array_equal(model.predict(first, np.full(n, 17.)), np.full(n, 17.))
    query = frame.iloc[:20].copy()
    query['clock'] = 4
    together = model.predict(query, np.full(len(query), 17.))
    separate = np.concatenate([model.predict(query.iloc[[i]], [17.]) for i in range(len(query))])
    np.testing.assert_allclose(together, separate)
    assert np.isfinite(together).all()
