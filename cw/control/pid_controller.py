import numpy as np
from collections import Sequence


# TODO: Write unitttest for control.

class PIDController:
    """
    Simple PID controller.

    :param float k_p: Proportional gain
    :param float k_i: Integral gain
    :param float k_d: Differential gain
    """
    def __init__(self, k_p, k_i, k_d):
        self.command = None
        self.output = np.empty((0,))
        self.integral = np.empty((0,))
        self.derivative = np.empty((0,))
        self.error = None
        self.time = None
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

    def step(self, time, value):
        """
        Runs a single step of the PID controller.

        :param time: Step time
        :param value: State value
        :return: Controller output. The same value is also stored <obj>.output.
        """

        # self.error will be None during the first step.
        if self.error is None:
            if isinstance(value, Sequence):
                # Get the length of values. And initialize the class members so they also have this size.
                length = len(value)
                # If no command has been set yet, set it to 0.
                if self.command is None:
                    self.command = np.zeros((length,))
                self.output = np.zeros((length,))
                self.integral = np.zeros((length,))
                self.derivative = np.zeros((length,))
            else:
                if self.command is None:
                    self.command = 0
                self.output = 0
                self.integral = 0
                self.derivative = 0

            # Calculate the error.
            self.error = value - self.command
            self.time = time

            # Calculate the controller output for this iteration only based on the proportional gain.
            self.output = self.error * self.k_p
        else:
            # This will run on all other iterations.
            error = self.command - value
            dt = time - self.time
            self.time = time

            self.integral += (self.error + error) / 2 * dt
            self.derivative = (error - self.error) / dt
            self.error = error

            self.output = error * self.k_p + self.integral * self.k_i + self.derivative * self.k_d

        # Return output.
        return self.output
