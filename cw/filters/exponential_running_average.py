# TODO: Write unittests for ExponentialRunningAverage.


class ExponentialRunningAverage:
    """
    Class used to calculate the exponential running average of a data stream.
    The type is not limited to floats. This class supports any type that
    supports basic arithmetic (subtraction, addition and multiplication).

    :param initial: Optional initial value.
    :param weight: Weight to give to new values being added.
    """

    def __init__(self, initial=None, weight=0.8):
        self.value = initial  #: The current value of the running average.
        self.weight = weight  #: The weight of the running average.

    def update(self, value):
        """
        Update the running average with a new entry.

        :param value: Value to be added.
        :return: The new value of the running average.
        """
        # If the value has not already been set, set it.
        if self.value is None:
            self.value = value
        else:
            # Calculate the new value.
            self.value = ((1-self.weight) * self.value + self.weight * value)
        return self.value

    def reset(self):
        """
        Reset the value to None.
        """
        self.value = None