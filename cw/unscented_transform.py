from math import sqrt

import numpy as np
from scipy.linalg import cholesky


class UnscentedTransform:
    def __init__(self, f, alpha, beta, k):
        self.f = f
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.l = None
        self.wm0 = None
        self.wmi = None
        self.wc0 = None
        self.wci = None

    def __call__(self, x, p):
        x = np.asarray(x).reshape((-1, 1))
        p = np.atleast_2d(np.asarray(p))
        n = len(x)

        if self.l is None:
            self.l = self.alpha**2 * (n + self.k) - n
            self.wm0 = self.l / (n + self.l)
            self.wc0 = self.l / (n + self.l) + 1 - self.alpha**2 + self.beta
            self.wmi = 1 / (2 * (n + self.l))
            self.wci = 1 / (2 * (n + self.l))

        # sqrt((L + lambda) * P)
        sqrtllp = sqrt(n + self.l) * cholesky(p, lower=False)

        # Construct sigma points and propagate
        sigmas = [x]
        ys = [np.asarray(self.f(x)).reshape((-1, 1))]
        for i in range(1, n + 1):
            delta = sqrtllp[:, i - 1].reshape((-1, 1))

            sigma = x + delta
            y = np.asarray(self.f(sigma)).reshape((-1, 1))
            sigmas.append(sigma)
            ys.append(y)

            sigma = x - delta
            y = np.asarray(self.f(sigma)).reshape((-1, 1))
            sigmas.append(sigma)
            ys.append(y)

        # Estimate mean and covariance
        mu = self.wm0 * ys[0]
        for y in ys[1:]:
            mu += self.wmi * y

        su = self.wc0 * ((ys[0] - mu) @ (ys[0] - mu).T)
        for y in ys[1:]:
            su += self.wci * ((y - mu) @ (y - mu).T)

        cu = self.wc0 * ((sigmas[0] - x) @ (ys[0] - mu).T)
        for sigma, y in zip(sigmas, ys[1:]):
            cu += self.wci * ((sigma - x) @ (y - mu).T)

        return mu.squeeze(), su, cu


if __name__ == '__main__':
    A = np.array([[1., -2.1],
                  [1.4, 4.1]])

    def f(x):
        return A @ x

    x0 = [1.3, -2]
    p0 = np.array([[7, 1.2],
                   [1.2, 8.4]])

    ut = UnscentedTransform(f, 1e-1, 2, (3 - 2))

    x1, sk, ck = ut(x0, p0)

    print(x1, A @ x0)
    print()

    print(sk, "\n")
    print(A @ p0 @ A.T, "\n")
    print(sk - A @ p0 @ A.T, "\n")
