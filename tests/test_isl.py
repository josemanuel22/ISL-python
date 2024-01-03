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
        y_k = torch.tensor([[1.0, 2.0, 3.1, 3.9]], dtype=torch.float32)
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
        y_hat = torch.tensor([[1.0, 2.0, 3.1, 3.9]], dtype=torch.float32)
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

if __name__ == "__main__":
    unittest.main()
