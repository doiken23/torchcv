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
		image_arr = np.flip(image_arr, 1).copy()
		GT_arr = np.flip(GT_arr, 1).copy()

        if self.vertical:
            if np.random.rand() < self.p:
                image_arr = np.flip(image_arr, 2).copy()
                GT_arr = np.flip(GT_arr,2).copy()

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
        image_arr = np.rot90(image_arr, n[0], (1,2)).copy()
        GT_arr = np.rot90(GT_arr, n[0], (1,2)).copy()

        return (img, target)
