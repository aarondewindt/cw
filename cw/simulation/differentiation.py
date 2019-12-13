from collections import deque
import numpy as np


class FiniteDifference:
    def __init__(self, step_size, max_order=4):
        self.step_size = step_size
        self.max_order = min(max_order, 4)
        self.f_deque = deque()
        self.fd_equations = [
            lambda f, h: (-1*f[1]+1*f[0])/(1*1.0*h**1),
            lambda f, h: (1*f[2]-4*f[1]+3*f[0])/(2*1.0*h**1),
            lambda f, h: (-2*f[3]+9*f[2]-18*f[1]+11*f[0])/(6*1.0*h**1),
            lambda f, h: (3*f[4]-16*f[3]+36*f[2]-48*f[1]+25*f[0])/(12*1.0*h**1)
        ]

    def differentiate(self, value):
        # Append value to function deque
        self.f_deque.appendleft(value)
        len_f_deque = len(self.f_deque)

        if len_f_deque == 1:
            # If this is out first value, return 0.
            return np.zeros(np.shape(value))
        elif len_f_deque > (self.max_order + 1):
            # If we have more items then we need for our derivative.
            # pop the extra one from the deque.
            self.f_deque.pop()

        # Calculate derivative and return.
        return self.fd_equations[len(self.f_deque) - 2](self.f_deque, self.step_size)
