import unittest
from cw.filters import IteratedExtendedKalmanFilter
import sympy as sp
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# TODO: Solve problems with iekf library


class TestFilters(unittest.TestCase):
    @unittest.skip("This are failing and will be fixed during the iekf library rewrite.")
    def test_iekf_single_state_no_input(self):
        # Create filter
        x1 = sp.symbols("x1")
        iekf = IteratedExtendedKalmanFilter(
            x=[x1],
            z=['z1'],
            u=None,
            f=sp.Matrix([
                -.3 * sp.cos(x1) ** 3
            ]),
            h=sp.Matrix([
                x1 ** 3
            ]),
            g=sp.Matrix([
                1
            ]),
            max_iterations=10,
            eps=1e-5,
        )
        # print(iekf)

        np.random.seed(0)
        # Generate test data by simulating dynamics
        data = iekf.sim(
            x_0=[5],
            u=xr.Dataset(coords={"t": np.arange(0, 10, 0.01)}),
            system_noise=[1.],
            system_bias=[0.],
            measurement_noise=[5.],
            measurement_bias=[0.],
        )

        # Run filter.
        data = iekf.filter(
            data=data,
            x_0=[10],
            p_0=np.diag([100.]),
            q=[[1.]],
            r=[[5.]]
        )

        print(data)

        plt.figure(3)
        data.x1.to_series().plot()
        data.x1_est.to_series().plot()
        plt.show()

        for x_name in iekf.x_names:
            x_mse = ((data[x_name].values - data[f"{x_name}_est"].values)**2).mean()
            # print(f"{x_name} mse: {x_mse}")
            self.assertGreaterEqual(0.15, x_mse)

    @unittest.skip("This are failing and will be fixed during the iekf library rewrite.")
    def test_iekf_single_state_one_input(self):
        # Create filter
        x1, u1 = sp.symbols("x1 u1")
        iekf = IteratedExtendedKalmanFilter(
            x=[x1],
            z=['z1'],
            u=[u1,],
            f=sp.Matrix([
                -.3 * sp.cos(x1) ** 3 + 0.5 * u1**5
            ]),
            h=sp.Matrix([
                u1 * x1**3
            ]),
            g=sp.Matrix([
                1 + 0.5 * u1**2
            ]),
            max_iterations=10,
            eps=1e-5,
        )
        # print(iekf)

        # Create input signal.
        t = np.arange(0, 10, 0.01)
        u1_values = np.sin(2*np.pi*t)
        u = xr.Dataset(
            data_vars={
                "u1": (('t',), u1_values)
            },
            coords={"t": t}
        )

        # Generate test data by simulating dynamics
        np.random.seed(0)
        data = iekf.sim(
            x_0=[5],
            u=u,
            system_noise=[1.],
            system_bias=[0.],
            measurement_noise=[5.],
            measurement_bias=[0.],
        )

        # Run filter.
        data = iekf.filter(
            data=data,
            x_0=[10],
            p_0=np.diag([100.]),
            q=[[1.]],
            r=[[5.]]
        )

        # print(data)
        # plt.figure(3)
        # data.x1.to_series().plot()
        # data.x1_est.to_series().plot()
        # plt.show()

        for x_name in iekf.x_names:
            x_mse = ((data[x_name].values - data[f"{x_name}_est"].values)**2).mean()
            # print(f"{x_name} mse: {x_mse}")
            self.assertGreaterEqual(0.15, x_mse)

    @unittest.skip("This are failing and will be fixed during the iekf library rewrite.")
    def test_iekf_two_states_no_input(self):
        x1, x2 = sp.symbols("x1 x2")
        iekf = IteratedExtendedKalmanFilter(
            x=[x1, x2],
            z=['z1'],
            u=None,
            f=sp.Matrix([
                x2*sp.cos(x1)**3,
                -x2
            ]),
            h=sp.Matrix([
                x1**2 + x2**2
            ]),
            g=sp.Matrix(np.eye(2)),
            max_iterations=100,
            eps=1e-10,
        )
        # print(iekf)

        # Generate test data by simulating dynamics
        np.random.seed(0)
        data = iekf.sim(
            x_0=[2, -3],
            u=xr.Dataset(coords={"t": np.arange(0, 10, 0.01)}),
            system_noise=[1., 1.],
            system_bias=[0., 0.],
            measurement_noise=[1.],
            measurement_bias=[0.],
        )

        # Run filter.
        data = iekf.filter(
            data=data,
            x_0=[10, 1],
            p_0=np.diag([100., 100.]),
            q=np.eye(2),
            r=[[1.]]
        )

        # print(data)
        #
        # plt.figure(3)
        # data.x1.to_series().plot()
        # data.x1_est.to_series().plot()
        # data.x2.to_series().plot()
        # data.x2_est.to_series().plot()
        #
        # plt.figure(4)
        # data.ikef_i_count.to_series().plot()
        # plt.show()

        for x_name in iekf.x_names:
            x_mse = ((data[x_name].values - data[f"{x_name}_est"].values)**2).mean()
            # print(f"{x_name} mse: {x_mse}")
            self.assertGreaterEqual(1.3, x_mse)

    @unittest.skip("This are failing and will be fixed during the iekf library rewrite.")
    def test_iekf_two_states_one_input(self):

        x1, x2, u1 = sp.symbols("x1 x2 u1")
        iekf = IteratedExtendedKalmanFilter(
            x=[x1, x2],
            z=['z1'],
            u=[u1],
            f=sp.Matrix([
                x2*sp.cos(x1)**3 + u1,
                x2 * u1 - x2
            ]),
            h=sp.Matrix([
                x1**2 + x2**2 * u1
            ]),
            g=sp.Matrix(np.eye(2)),
            max_iterations=100,
            eps=1e-10,
        )
        # print(iekf)

        # Create input signal.
        t = np.arange(0, 10, 0.01)
        u1_values = 3 * np.sin(2*np.pi*t)
        u = xr.Dataset(
            data_vars={
                "u1": (('t',), u1_values)
            },
            coords={"t": t}
        )

        # Generate test data by simulating dynamics
        np.random.seed(0)
        data = iekf.sim(
            x_0=[2, -3],
            u=u,
            system_noise=[1., 1.],
            system_bias=[0., 0.],
            measurement_noise=[1.],
            measurement_bias=[0.],
        )

        # plt.figure(3)
        # data.x1.to_series().plot()
        # data.x2.to_series().plot()
        # plt.show()
        # return

        # Run filter.
        data = iekf.filter(
            data=data,
            x_0=[10, 1],
            p_0=np.diag([100., 100.]),
            q=np.eye(2),
            r=[[1.]]
        )

        print(data)

        plt.figure(3)
        data.x1.to_series().plot()
        data.x1_est.to_series().plot()
        data.x2.to_series().plot()
        data.x2_est.to_series().plot()

        plt.figure(4)
        data.ikef_i_count.to_series().plot()
        plt.show()

        for x_name in iekf.x_names:
            x_mse = ((data[x_name].values - data[f"{x_name}_est"].values)**2).mean()
            # print(f"{x_name} mse: {x_mse}")
            self.assertGreaterEqual(0.2, x_mse)

    @unittest.skip("This are failing and will be fixed during the iekf library rewrite.")
    def test_iekf_two_states_two_input(self):
        x1, x2, u1, u2 = sp.symbols("x1 x2 u1 u2")
        iekf = IteratedExtendedKalmanFilter(
            x=[x1, x2],
            z=['z1'],
            u=[u1, u2],
            f=sp.Matrix([
                x2*sp.cos(x1)**3 + u1,
                u2 - x2
            ]),
            h=sp.Matrix([
                x1**2 + x2**2 * u1
            ]),
            g=sp.Matrix(np.eye(2)),
            max_iterations=100,
            eps=1e-10,
        )
        print(iekf)
        iekf.print_latex()

        # Create input signal.
        t = np.arange(0, 10, 0.01)
        u1_values = 3 * np.sin(2*np.pi*t)
        u2_values = 6 * np.cos(6*np.pi*t)
        u = xr.Dataset(
            data_vars={
                "u1": (('t',), u1_values),
                "u2": (('t',), u2_values)
            },
            coords={"t": t}
        )

        # Generate test data by simulating dynamics
        np.random.seed(0)
        data = iekf.sim(
            x_0=[2.0, -3.0],
            u=u,
            system_noise=[2., 1.],
            system_bias=[0., 0.],
            measurement_noise=[3.],
            measurement_bias=[0.],
        )

        # plt.figure(3)
        # data.x1.to_series().plot()
        # data.x2.to_series().plot()
        # plt.show()
        # return

        # Run filter.
        data = iekf.filter(
            data=data,
            x_0=[10.0, 1.0],
            p_0=np.diag([100., 100.]),
            q=np.eye(2),
            r=[[1.]],
            verbose=False
        )

        # print(data)
        # plt.figure(figsize=(10, 2))
        #
        # plt.tight_layout()

        plt.figure(figsize=(10, 6))
        plt.subplot(311)
        plt.title("IEKF unit-test inputs")
        data.u1.to_series().plot(label="$u_1$")
        data.u2.to_series().plot(label="$u_2$", style="--")
        plt.legend(loc=5)
        plt.subplot(312)
        plt.title("IEKF unit-test measurement")
        data.z1.to_series().plot(label="$z_1$")
        # plt.legend()
        plt.ylabel("$z_1$")
        plt.subplot(313)
        plt.title("IEKF unit-test result")
        data.x1.to_series().plot(label="$x_1$", linewidth=0.75)
        data.x1_est.to_series().plot(label="$x_{1_{estimate}}$", style="--")
        data.x2.to_series().plot(label="$x_2$", linewidth=2.0)
        data.x2_est.to_series().plot(label="$x_{2_{estimate}}$", style=":")
        plt.legend(loc=5)
        plt.tight_layout()

        # plt.figure(4)
        # data.iekf_i_count.to_series().plot()
        plt.show()

        # for x_name in iekf.x_names:
        #     x_mse = ((data[x_name].values - data[f"{x_name}_est"].values)**2).mean()
        #     # print(f"{x_name} mse: {x_mse}")
        #     self.assertGreaterEqual(1.155, x_mse)


if __name__ == '__main__':
    unittest.main()
