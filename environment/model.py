import numpy as np


class Base():
    def __init__(self, position: tuple, center: tuple, size: tuple):
        from .image_utils import to_bounds
        self.__position = position
        self.__center = center
        self.__size = size
        self.__bounds = to_bounds(self.__position, self.__size)

    @property
    def position(self):
        return self.__position

    @property
    def center(self):
        return self.__center

    @property
    def size(self):
        return self.__size

    @property
    def bounds(self):
        return self.__bounds


class Image(Base):
    def __init__(self, data: np.ndarray, position=(0, 0)):
        assert (data is not None), "data must not be empty"
        from .image_utils import div, sum, to_int
        size = data.shape[::-1][-2:]
        center = to_int(sum(div(size, (2, 2)), position))
        super().__init__(position, center, size)
        self.__data = data
        self.__channels = data.shape[2]
        self.__shape = data.shape

    @property
    def data(self):
        return self.__data

    @property
    def channels(self):
        return self.__channels

    @property
    def shape(self):
        return self.__shape


class View(Base):
    def __init__(self, position, center, size, scale):
        super().__init__(position, center, size)
        self.__scale = scale

    @property
    def scale(self):
        return self.__scale


class Step:
    def __init__(self, name=None, move=(0, 0), scale=1.0):
        super().__init__()
        self.name = name
        self.scale = scale
        self.move = move
