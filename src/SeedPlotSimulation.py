import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
from sklearn.datasets import make_blobs

@dataclass
class SimulationResults:
    n_found: int
    path_length: float
    found_plants: List[Point]
    detection_rate: float

class SeedPlotSimluation:
    def __init__ (self,
                  width: float = 10.0,
                  height: float = 10.0,
                  n_plants: int = 20,
                  distribution: str = 'uniform',
                  distribution_params: dict = None,
                  detection_params: float = 0.2):
        
        self.width = width
        self.height = height
        self.n_plants = n_plants
        self.distribution = distribution
        self.distribution_params = distribution_params or {}
        self.detection_params = detection_params

        self.plants = self._generate_plants()
        self.plant_buffers = self._generate_plant_buffers()
    
    def _generate_plants(self) -> List[Point]:
        """
        Generate plant locations based on a user specified distribution type.
        """
        distribution_methods = {
            'uniform': self._generate_uniform_plants,
            'clustered': self._generate_clustered_plants,
            'gradient': self._generate_gradient_plants
        }

        if self.distribution not in distribution_methods:
            raise ValueError("Unknown distribution, please indicate: uniform, clustered, or gradient")
        
        return distribution_methods[self.distribution]()
    
    def _generate_uniform_plants(self) -> List[Point]:
        """
        Generate plant locations based on a uniform distribution
        """
        x_coordinates = np.random.uniform(0, self.width, self.n_plants)
        y_coordinates = np.random.uniform(0, self.height, self.n_plants)

        return [Point(x, y) for x, y in zip(x_coordinates, y_coordinates)]
    
    def _generate_clustered_plants(self) -> List[Point]:
        """
        Generate plant locations based on a Isotropic Gaussian Cluster distribution
        https://scikit-learn.org/1.5/modules/generated/sklearn.datasets.make_blobs.html
        """
        n_clusters = self.distribution_params.get('n_clusters', 3)
        n_cluster_std = self.distribution_params.get('n_cluster_std', 1.0)


        cluster_data, labels = make_blobs(n_samples=100, cluster_std=n_cluster_std, centers=n_clusters, n_features=2, random_state=1)
        
        x_coordinates = [float(x[0]) for x in cluster_data]
        y_coordinates = [float(y[1]) for y in cluster_data]

        return [Point(x, y) for x, y in zip(x_coordinates, y_coordinates)]
    
    def _generate_gradient_plants(self) -> List[Point]:
        """
        Generate plants with density following a probability gradient.
        
        Attempts to place points in random points and determines if they are placed
        with a probability weight influenced by "direction" and "steepness"

        "direction" is if the gradient goes from left-right or up-down
        "steepness" is how heavily points far from their "direction" are punished
        """
        ## Copied directly from stackoverflow, may need to find a better solution
        direction = self.distribution_params.get('direction', 'x')
        steepness = self.distribution_params.get('steepness', 1.0)
        
        points = []
        while len(points) < self.n_plants:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            
            if direction == 'x':
                prob = np.exp(-steepness * x / self.width)
            elif direction == 'y':
                prob = np.exp(-steepness * y / self.height)
            else:
                prob = np.exp(-steepness * (x + y) / (self.width + self.height))
                
            if np.random.random() < prob:
                points.append(Point(x, y))
        return points
    
    def _generate_plant_buffers(self):
        return
    
    def _create_plot_plants(self):
        x_points = [point.x for point in self.plants]
        y_points = [point.y for point in self.plants]

        plt.figure(figsize=(self.width, self.height))  # Optional: Set figure size
        plt.scatter(x_points, y_points, c='red', label='Points')  # Scatter plot
        plt.title("Scatter Plot of Shapely Points")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.axhline(0, color='gray', linewidth=0.5)  # Optional: Add horizontal line
        plt.axvline(0, color='gray', linewidth=0.5)  # Optional: Add vertical line
        plt.grid(True)  # Optional: Add grid
        plt.legend()
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        
        plt.show()