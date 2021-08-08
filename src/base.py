
class BaseEstimator:
    def __str__(self):
        props = ', '.join([f"{prop[0]}={prop[1]}" for prop in self.__dict__.items() if not prop[0].startswith("_")])
        return f"{self.__class__.__name__}({props})"

    def __repr__(self):
        return self.__str__()
