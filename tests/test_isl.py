import sys
sys.path.append('../ISL-python')

import ISL.isl
import torch
import unittest
class TestFunctions(unittest.TestCase):

    def test_sigmoid(self):
        y_hat = torch.tensor(0.5)
        y = torch.tensor(1.0)
        expected = torch.sigmoid((y - y_hat) * 10.0)
        self.assertTrue(torch.isclose(ISL.isl._sigmoid(y_hat, y), expected))

    def test_sigmoid_negative(self):
        y_hat = torch.tensor(1.0)
        y = torch.tensor(0.5)
        expected = torch.sigmoid((y - y_hat) * 10.0)
        self.assertTrue(torch.isclose(ISL.isl._sigmoid(y_hat, y), expected))

    def test_psi_m(self):
        y = torch.tensor(1.0)
        m = torch.tensor(1.0)
        stddev = 0.1
        expected = torch.exp(-0.5 * ((y - m) / stddev) ** 2)
        self.assertTrue(torch.isclose(ISL.isl.psi_m(y, m), expected))

    def test_phi(self):
        y_k = torch.tensor([0.5, 0.6, 0.7])
        y_n = torch.tensor(0.65)
        expected = torch.sum(ISL.isl._sigmoid(y_k, y_n))
        self.assertTrue(torch.isclose(ISL.isl.phi(y_k, y_n), expected))

    def test_gamma(self):
        y_k = torch.tensor([0.5, 0.6, 0.7])
        y_n = torch.tensor(0.65)
        m = 1
        expected = torch.zeros(len(y_k) + 1)
        expected[m] = ISL.isl.psi_m(ISL.isl.phi(y_k, y_n), m)
        self.assertTrue(torch.all(torch.eq(ISL.isl.gamma(y_k, y_n, m), expected)))

    def test_gamma_approx(self):
        y_k = torch.tensor([1.0, 2.0, 3.1, 3.9], dtype=torch.float32)
        y_n = torch.tensor(3.6, dtype=torch.float32)
        m = 3
        expected = torch.tensor([0.0, 0.0, 0.0, 0.92038, 0.0], dtype=torch.float32)
        tol = 1e-5  # Assuming 'tol' is defined somewhere in your tests

        result = ISL.isl.gamma(y_k, y_n, m)
        self.assertTrue(torch.allclose(result, expected, atol=tol))

    def test_generate_a_k(self):
        y_hat = torch.tensor([0.5, 0.6, 0.7])
        y = torch.tensor(0.65)
        K = len(y_hat)
        expected = torch.sum(torch.stack([ISL.isl.gamma(y_hat, y, k) for k in range(K)]), dim=0)
        self.assertTrue(torch.all(torch.eq(ISL.isl.generate_a_k(y_hat, y), expected)))

    def test_generate_a_k_approx(self):
        y_hat = torch.tensor([1.0, 2.0, 3.1, 3.9], dtype=torch.float32)
        y = torch.tensor(3.6, dtype=torch.float32)
        expected = torch.tensor([0.0, 0.0, 0.0, 0.92038, 0.0], dtype=torch.float32)
        tol = 1e-5  # Assuming 'tol' is defined somewhere in your tests

        result = ISL.isl.generate_a_k(y_hat, y)
        self.assertTrue(torch.allclose(result, expected, atol=tol))

    def test_scalar_diff(self):
        q = torch.tensor([0.2, 0.3, 0.5])
        K = len(q)
        expected = torch.sum((q - 1/(K+1)) ** 2)
        self.assertTrue(torch.isclose(ISL.isl.scalar_diff(q), expected))

    def test_scalar_diff_negative_values(self):
        q = torch.tensor([-0.2, 0.3, -0.5])
        K = len(q)
        expected = torch.sum((q - 1/(K+1)) ** 2)
        self.assertTrue(torch.isclose(ISL.isl.scalar_diff(q), expected))

    # Mock model for testing
    def mock_model(self, x):
        # This mock model can be adjusted to return predictable values
        return x * 2

    def test_get_window_of_Ak(self):
        mock_data = [torch.tensor([2.0]), torch.tensor([4.0])]
        K = 2
        result = ISL.isl.get_window_of_Ak(self.mock_model, mock_data, K)
        # Expected result based on the mock_model and mock_data
        expected = [2, 1, 0]
        self.assertEqual(result, expected)

    def test_convergence_to_uniform(self):
        uniform_distribution = [25, 25, 25, 25]  # Uniform
        non_uniform_distribution = [5, 5, 20, 70]  # Non-uniform
        approx_uniform_distribution = [20, 30, 20, 30]  # approx-uniform
        limit_uniform_distribution_negative = [15, 35, 20, 30]  # limit-uniform-
        limit_uniform_distribution_positive = [17, 33, 20, 30]  # limit-uniform+
        self.assertTrue(ISL.isl.convergence_to_uniform(uniform_distribution))
        self.assertFalse(ISL.isl.convergence_to_uniform(non_uniform_distribution))
        self.assertTrue(ISL.isl.convergence_to_uniform(approx_uniform_distribution))
        self.assertFalse(ISL.isl.convergence_to_uniform(limit_uniform_distribution_negative))
        self.assertFalse(ISL.isl.convergence_to_uniform(limit_uniform_distribution_positive))

    def test_get_better_K(self):
        mock_data = [torch.tensor([100.0]), torch.tensor([100.0]), torch.tensor([100.0])]
        hparams = {'max_k': 100}
        expected_K = 2  # Expected K value for these inputs
        result_K = ISL.isl.get_better_K(self.mock_model, mock_data, 2, hparams)
        self.assertEqual(result_K, expected_K)

if __name__ == "__main__":
    unittest.main()
