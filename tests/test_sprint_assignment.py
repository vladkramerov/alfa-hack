import json
import pickle
import unittest

import numpy as np
import pandas as pd

from sprint_assignment import GlobalEntityAssignment


def frame(years, areas, built=None, elevation=None):
    n = len(years)
    return pd.DataFrame({'Year_Factor': years, 'State_Factor': ['S']*n,
        'building_class': ['Commercial']*n, 'facility_type': ['Office']*n,
        'floor_area': areas, 'year_built': built if built is not None else [1950.]*n,
        'ELEVATION': elevation if elevation is not None else [0.]*n,
        'energy_star_rating': [50.]*n})


def baseline(X):
    return np.full(len(X), 10.)


class GlobalAssignmentTests(unittest.TestCase):
    def test_joint_assignment_resolves_local_nearest_collision_without_donor_reuse(self):
        model = GlobalEntityAssignment().fit(frame([1,1],[100.,104.]), [100.,300.], baseline)
        q = frame([2,2],[101.,101.5])
        # Both local nearest donors are100; joint min-cost allocation is100 and104.
        np.testing.assert_allclose(model.predict(q), [41.5,111.5])
        self.assertEqual(model.last_prediction_audit_['donor_positions'], [0,1])
        self.assertEqual(model.last_prediction_audit_['assigned_rows'], 2)
        # A single row may use another donor: cohort dependence is intentional.
        np.testing.assert_allclose(model.predict(q.iloc[[1]]), [41.5])

    def test_exact_links_reserve_donor_and_query_duplicates_stay_baseline(self):
        model = GlobalEntityAssignment().fit(frame([1,1],[100.,104.]), [100.,300.], baseline)
        q = frame([2,2],[100.,101.5])
        np.testing.assert_allclose(model.predict(q), [10.,111.5])
        self.assertEqual(model.last_prediction_audit_['reserved_donor_profiles'], 1)
        np.testing.assert_array_equal(model.predict(frame([2,2],[101.,101.])), [10.,10.])
        self.assertEqual(model.last_prediction_audit_['duplicate_query_rows'], 2)

    def test_latest_global_training_year_only_strictpast_and_block_gates(self):
        model = GlobalEntityAssignment().fit(frame([1,3,5],[100.,104.,108.]), [100.,300.,500.], baseline)
        q = frame([1,2,4,5],[101.,101.,105.,105.])
        np.testing.assert_allclose(model.predict(q), [10.,41.5,111.5,10.+.35*np.exp(-.25)*(300.-10.)])
        self.assertEqual(model.last_prediction_audit_['donor_years'], [None,1.,3.,3.])
        off = frame([2],[101.]); off['State_Factor'] = 'OTHER'
        np.testing.assert_array_equal(model.predict(off), [10.])
        # A far-away latest-year donor must not trigger fallback to a close older year.
        latest_only=GlobalEntityAssignment().fit(frame([1,3],[100.,1000.]),[100.,999.],baseline)
        np.testing.assert_array_equal(latest_only.predict(frame([4],[101.])),[10.])

    def test_cost_gates_missing_penalties_and_optional_elevation(self):
        model = GlobalEntityAssignment().fit(frame([1],[100.]), [100.], baseline)
        q = frame([2,2,2,2,2],[101.,106.,100.,100.,101.],
                  built=[1952.,1950.,1971.,0.,1950.], elevation=[10.,0.,0.,np.nan,90.])
        cost = model._cost_matrix(q, [0,1,2,3,4], [0])[:,0]
        self.assertAlmostEqual(cost[0], np.log(1.01)/.05+2/20+10/100)
        self.assertFalse(np.isfinite(cost[1])); self.assertFalse(np.isfinite(cost[2]))
        self.assertEqual(cost[3], 1.)
        self.assertFalse(np.isfinite(cost[4]))  # Legal gates, but cost>1 is forbidden before assignment.
        np.testing.assert_allclose(model.predict(q.iloc[[3]]), [41.5])
        no_elevation = frame([1],[100.]).drop(columns=['ELEVATION'])
        optional = GlobalEntityAssignment().fit(no_elevation,[100.],baseline)
        query = frame([2],[100.],built=[1951.]).drop(columns=['ELEVATION'])
        np.testing.assert_allclose(optional.predict(query), [41.5])

    def test_full_dummy_matrix_cap_and_ambiguous_training_profiles(self):
        model = GlobalEntityAssignment(max_assignment_cells=5).fit(frame([1,1],[100.,104.]),[100.,300.],baseline)
        np.testing.assert_array_equal(model.predict(frame([2,2],[101.,101.5])), [10.,10.])
        self.assertEqual(model.last_prediction_audit_['skipped_blocks'][0]['cells'],8)
        self.assertEqual(model.last_prediction_audit_['skipped_blocks'][0]['reason'],'matrix_cell_cap')
        duplicate = GlobalEntityAssignment().fit(frame([1,1,1],[100.,100.,104.]),[100.,999.,300.],baseline)
        self.assertEqual(duplicate.audit_['rejected_ambiguous_profiles'],1)
        np.testing.assert_array_equal(duplicate.predict(frame([2],[100.])), [10.])

    def test_frozen_rating_transfer_pickle_and_permutation_keyed_equivalence(self):
        X = frame([1,1],[100.,104.]); y=[100.,300.]
        state={'rating_beta': .5, 'group_rating_betas': {'Office': .5}}
        model = GlobalEntityAssignment().fit(X,y,baseline,state)
        q = frame([2,2],[101.,101.5]); q['id']=[99,1]; q['energy_star_rating']=[20.,80.]
        pred=model.predict(q)
        self.assertGreater(pred[0],41.5); self.assertLess(pred[1],111.5)
        state['rating_beta']=999.; state['group_rating_betas']['Office']=999.
        restored=pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(q),pred)
        reverse=GlobalEntityAssignment().fit(X.iloc[::-1],y[::-1],baseline,model.transport_state_)
        np.testing.assert_array_equal(reverse.predict(q.iloc[::-1])[::-1],pred)
        np.testing.assert_array_equal(model.predict(q),pred)
        json.dumps(model.configuration()); json.dumps(model.audit_); json.dumps(model.last_prediction_audit_)

    def test_invalid_target_transport_and_baseline_are_rejected(self):
        with self.assertRaises(ValueError): GlobalEntityAssignment().fit(frame([1],[100.]),[-1.])
        with self.assertRaises(ValueError):
            GlobalEntityAssignment().fit(frame([1],[100.]),[100.],transport_state={'rating_beta':np.nan})
        model=GlobalEntityAssignment().fit(frame([1],[100.]),[100.],lambda q: np.full(len(q),np.nan))
        with self.assertRaises(ValueError): model.predict(frame([2],[101.]))


if __name__=='__main__':
    unittest.main()
