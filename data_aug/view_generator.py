import numpy as np

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class WatermarkViewGenerator(object):

    def __init__(self, base_transform, n_views):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [transform(x) for transform in self.base_transform]