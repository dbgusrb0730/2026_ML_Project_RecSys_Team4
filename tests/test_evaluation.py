import unittest
import numpy as np
from scipy.sparse import csr_matrix
from evaluation import MetricAccumulator, top_items, evaluate_model
from memory import ImprovedUserKNN


class EvaluationTests(unittest.TestCase):
    def test_empty_relevance_included_in_primary(self):
        known = csr_matrix((2, 12))
        test = csr_matrix(([5., 2.], ([0,1],[0,1])),shape=(2,12))
        scores = np.tile(np.arange(12,0,-1),(2,1))
        acc = MetricAccumulator()
        acc.add(scores,np.array([0,1]),known,test,rating_scores=np.full((2,12),3.))
        result = acc.result()
        self.assertAlmostEqual(result['rmse'], np.sqrt(2.5))
        self.assertAlmostEqual(result['all_users']['f1'], 1/11)
        self.assertAlmostEqual(result['relevant_users']['f1'], 2/11)
        self.assertEqual(result['n_users_all'],2)
        self.assertEqual(result['n_users_relevant'],1)

    def test_mask_and_tie(self):
        scores = np.array([100.,2.,2.,2.,1.])
        eligible = np.array([False,True,True,True,False])
        np.testing.assert_array_equal(top_items(scores,eligible,2), [1,2])
        np.testing.assert_array_equal(top_items(scores,eligible,10), [1,2,3])
        self.assertEqual(len(top_items(scores,np.zeros(5,bool),10)),0)

    def test_known_and_candidate_exclusion(self):
        scores = np.array([[10.,9.,8.]])
        known = csr_matrix(([1.],([0],[0])),shape=(1,3))
        test = csr_matrix(([5.],([0],[2])),shape=(1,3))
        result = evaluate_model(lambda users:scores, np.array([0]),known,test,np.array([True,False,True]),False)
        self.assertEqual(result['all_users']['recall'],1.)
        self.assertIsNone(result['rmse'])
        self.assertEqual(result['n_test_ratings'],1)

    def test_bias_ablation_without_neighbors(self):
        u=np.array([0,0,1,1]);i=np.array([0,1,2,3]);r=np.array([5.,4.,5.,4.])
        model=ImprovedUserKNN(k=1).fit(u,i,r,(2,12))
        np.testing.assert_allclose(model.scores([0,1]),model.baseline_scores([0,1]))

    def test_cross_similarity_and_neighbor_mask(self):
        # Target-item column must never enter the similarity matrix.
        u=np.array([0,1,1,2,2]);i=np.array([0,0,2,0,3]);r=np.array([5.,5.,4.,5.,5.])
        su=np.array([0,1,2]);si=np.array([0,0,0]);sr=np.array([5.,5.,5.])
        model=ImprovedUserKNN(k=2).fit(u,i,r,(3,12),similarity_data=(su,si,sr),neighbor_mask=np.array([False,True,False]))
        self.assertEqual(model.A[:,2:].nnz,0)
        rating,vote=model.score_pair([0])
        self.assertGreater(vote[0,2],0)
        self.assertLess(vote[0,3],0)
        self.assertTrue(np.isfinite(rating).all())


if __name__ == '__main__':
    unittest.main()
