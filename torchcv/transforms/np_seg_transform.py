import numpy as np

class NPSegRandomFlip(object):
    """
    flip transform for numpy array.
    transform img and target.
    """
    def __init__(self, p=0.5, horizontal=True, vertical=True):
        self.p = p
        self.horizontal = horizontal
        self.vertical = vertical

    def __call__(self, data):

        img, target = data

        if self.horizontal:
            if np.random.rand() < self.p:
                img = np.flip(img, 1).copy()
                target = np.flip(target, 1).copy()

        if self.vertical:
            if np.random.rand() < self.p:
                img = np.flip(img, 2).copy()
                target = np.flip(target, 2).copy()

        return (img, target)

class NPSegRandomRotate(object):
    """
    rotation transform for numpy array.
    transform img and target.
    """
    def __init__(self):
        pass

    def __call__(self, data):

        img, target = data

        n = np.random.choice([0, 1, 2, 3])
        img = np.rot90(img, n, (1,2)).copy()
        target = np.rot90(target, n, (1,2)).copy()

        return (img, target)
