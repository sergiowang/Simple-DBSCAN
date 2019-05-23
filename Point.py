import numpy as np

class Point(object):
    def __init__(self, **kwargs):
        if 'position' not in kwargs:
            raise ValueError('Point must have its position!')
        self.position = np.array(kwargs['position'])
        self.neighboursDirect = []
        self.type = 'base'
        self.visited = False
        self.cluster = None

    def set_cluster(self, cluster):
        self.cluster = cluster



if __name__ == '__main__':
    p = Point(position = [1, 2, 3, 4])
    