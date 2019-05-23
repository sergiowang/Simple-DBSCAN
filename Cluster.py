from Point import Point
import pandas as pd
import numpy as np
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
class Cluster(object):
    """
    Implement dbscan algorithm 
    """
    def __init__(self, eps, minPts):
        self.eps = eps
        self.minPts = minPts
        self.points = []

    @staticmethod
    def dis(point1, point2):
        return sqrt(sum((point1.position - point2.position) ** 2))

    def form_points(self, data):
        for p in data.values.tolist():
            self.points.append(Point(position = p))

    def find_neighbour(self, point):
        neighbours = []
        for other in self.points:
            if self.dis(point, other) <= self.eps:
                neighbours.append(other)
        return neighbours

    def cluster(self):
        currentCluster = 0
        for p in self.points:
            # determine if the point can be a core and form a new cluster
            if p.cluster is not None:
                continue
            neighbours = self.find_neighbour(p)
            if len(neighbours) < self.minPts:
                p.type = 'noise'
                continue
            p.type = 'core'
            currentCluster += 1
            p.set_cluster(currentCluster)
            # spread this cluster via its neighbours
            for neighbour in neighbours:
                if neighbour.cluster is not None:
                    continue
                if neighbour.type == 'noise':
                    neighbour.type = 'reachable'
                neighbour.set_cluster(currentCluster)
                spreads = self.find_neighbour(neighbour)
                if len(spreads) <= self.minPts:
                    # if this reachable point is not core, it won't spread its cluster lable to its neighbours
                    continue
                else:
                    # this is a core point, spread its cluster lable to its neighbours
                    neighbour.type = 'core'
                    neighbours += spreads
        [p.set_cluster(-1) for p in self.points if p.type == 'noise']

    def attach_result(self, data):
        cluster = []
        for point in self.points:
            cluster.append(point.cluster)
        data['cluster'] = pd.Series(cluster, index = data.index)

    def fit(self, data):
        self.form_points(data)
        self.cluster()
        self.attach_result(data)
        return data
    
    @staticmethod
    def plot2D(data):
        #cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
        ax = sns.scatterplot(x = "x", y="y",
                            hue = "cluster",
                            #palette = cmap, 
                            sizes=(10, 200),
                            data = data)
        plt.show()





if __name__ == '__main__':
    testData1 = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 1000)
    testData2 = np.random.multivariate_normal([5, 5], [[1.5, 0], [0, 0.5]], 1000)
    testData = np.concatenate((testData1, testData2), axis=0)
    testData = pd.DataFrame(testData, columns = ['x', 'y'])
    dbscanCluster = Cluster(0.5, 50)
    result = dbscanCluster.fit(testData)
    print('clusters:', pd.unique(result['cluster']))
    dbscanCluster.plot2D(result)
    result.to_csv('result.csv', index = False)

