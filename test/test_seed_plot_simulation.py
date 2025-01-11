import pytest
from src.SeedPlotSimulation import SeedPlotSimluation
from shapely.geometry import Point
from typing import List
from scipy.stats import ks_1samp, uniform
import numpy as np
from sklearn.cluster import KMeans

def test_initialization_defaults():
    ## Data initializaiton
    sim = SeedPlotSimluation()
    x_points = [point.x for point in sim.plants]
    y_points = [point.y for point in sim.plants]

    ## Confirming class parameters are maintained correctly
    assert sim.width == 10.0
    assert sim.height == 10.0
    assert sim.n_plants == 20
    assert sim.distribution == 'uniform'
    assert sim.distribution_params == {}
    assert sim.detection_radii == 0.2

    assert isinstance(sim.plants, list)
    assert isinstance(sim.plants[0], Point)
    assert isinstance(x_points[0], float)
    assert isinstance(y_points[0], float)


def test_generate_uniform_plants():
    ## Testing Null Hypothesis of Uniformity with the Kolmogorov-Smirnov test
    ## https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    # Perform KS test for uniformity along X and Y axes
    n_trials = 100

    p_values_x = []
    p_values_y = []

    for _ in range(n_trials):
        ## Data initializaiton
        sim = SeedPlotSimluation(n_plants = 1000, distribution = 'uniform')
        x_points = [point.x for point in sim.plants]
        y_points = [point.y for point in sim.plants]
        
        ks_stat_x, p_value_x = ks_1samp(x_points, uniform(loc=0, scale=sim.width).cdf)
        ks_stat_y, p_value_y = ks_1samp(y_points, uniform(loc=0, scale=sim.height).cdf)

        p_values_x.append(p_value_x)
        p_values_y.append(p_value_y)
    
    ## Data initializaiton
    sim = SeedPlotSimluation(distribution = 'uniform')
    x_points = [point.x for point in sim.plants]
    y_points = [point.y for point in sim.plants]

    ## Confirming class parameters are maintained correctly
    assert sim.width == 10.0
    assert sim.height == 10.0
    assert sim.n_plants == 20
    assert sim.distribution == 'uniform'
    assert sim.distribution_params == {}
    assert sim.detection_radii == 0.2

    assert isinstance(sim.plants, list)
    assert isinstance(sim.plants[0], Point)
    assert isinstance(x_points[0], float)
    assert isinstance(y_points[0], float)

    #KS test H0 = dataset comes from the target distribution (is from a uniform distribution)
    assert np.mean(p_values_x) > 0.05
    assert np.mean(p_values_y) > 0.05

## Not messing around with statistical tests for testing how clustered data is currently
## Maybe in the future
def test_generate_clustered_plants():
    ## Data initializaiton
    sim = SeedPlotSimluation(distribution = 'clustered')
    x_points = [point.x for point in sim.plants]
    y_points = [point.y for point in sim.plants]

    ## Confirming class parameters are maintained correctly
    assert sim.width == 10.0
    assert sim.height == 10.0
    assert sim.n_plants == 20
    assert sim.distribution == 'clustered'
    assert sim.distribution_params == {}
    assert sim.detection_radii == 0.2

    assert isinstance(sim.plants, list)
    assert isinstance(sim.plants[0], Point)
    assert isinstance(x_points[0], float)
    assert isinstance(y_points[0], float)


def test_generate_gradient_plants():
    ## Data initializaiton
    sim = SeedPlotSimluation(distribution = 'gradient')
    x_points = [point.x for point in sim.plants]
    y_points = [point.y for point in sim.plants]

    ## Calculating difference between sides of gradient wall
    gradient_left = [point for point in sim.plants if point.x < (sim.width/2)]
    gradient_right = [point for point in sim.plants if point not in gradient_left]

    ## Confirming class parameters are maintained correctly
    assert sim.width == 10.0
    assert sim.height == 10.0
    assert sim.n_plants == 20
    assert sim.distribution == 'gradient'
    assert sim.distribution_params == {}
    assert sim.detection_radii == 0.2

    assert isinstance(sim.plants, list)
    assert isinstance(sim.plants[0], Point)
    assert isinstance(x_points[0], float)
    assert isinstance(y_points[0], float)

    assert len(gradient_left) > len(gradient_right)