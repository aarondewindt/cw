from abc import ABCMeta


# TODO: Write unittests for singleton.


class Singleton(metaclass=ABCMeta):
    """
    Base class for all singleton objects. These are object of which only 
    one instance of its type may exist at a time.
    """
    instance = None

    def __new__(cls, *args, **kwargs):
        """Returns existing instance of the derived if it exists already, otherwise it creates it."""
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __repr__(self):
        return "Singleton"

    def __str__(self):
        return "Singleton"

    def __call__(self, *args, **kwargs):
        return self


class ValidType(Singleton):
    def __repr__(self):
        return "Valid"


Valid = ValidType()
