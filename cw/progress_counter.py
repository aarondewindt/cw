from time import time


# TODO: Write unittests for progress counter
# TODO: Write documentation for progress counter.


class ProgressCounter:
    def __init__(self, initial=0, total=None, weight=0.8, update_period=1):
        self.count = initial
        self.total = total
        self.weight = weight
        self.update_period = update_period
        self.rate = 0

        self.last_update_time = None
        self.last_update_count = self.count

    def update(self, n=1):
        self.count += n
        if self.last_update_time is None:
            self.last_update_time = time()
        else:
            current_time = time()
            if current_time - self.last_update_time >= self.update_period:
                rate = (self.count - self.last_update_count) / (current_time - self.last_update_time)
                self.rate = (1-self.weight) * self.rate + self.weight * rate
                self.last_update_count = self.count
                self.last_update_time = current_time

    @property
    def eta(self):
        if self.total:
            if self.rate != 0:
                return (self.total - self.count) / self.rate
            else:
                return float("inf")
