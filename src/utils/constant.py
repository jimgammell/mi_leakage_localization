class Constant:
    def __init__(self, value):
        self.value = value
    def __get__(self, *args):
        return self.value
    def __set__(self, *args):
        raise AttributeError('Constant cannot be modified at runtime.')