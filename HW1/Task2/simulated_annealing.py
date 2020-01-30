import math
import random
import matplotlib.pyplot as plt
import numpy as np
import geopy.distance

from Task2 import visualization


def to_distance_matrix(coords):
    """
    Convert coordinates of cities into distance matrix
    :param coords: cities' coordinates
    :return: distance matrix
    """
    matrix = np.zeros((coords.shape[0], coords.shape[0]))
    for i in range(coords.shape[0]):
        for j in range(coords.shape[0]):
            matrix[i][j] = geopy.distance.vincenty((coords[i][0], coords[i][1]), (coords[j][0], coords[j][1])).km
    return matrix


def find_nearest_neighbours(dist_matrix):
    """
    Used to find the initial solution based on nearest neighbours search
    :param dist_matrix: matrix of distances between cities
    :return: initial solution
    """
    node = random.randrange(len(dist_matrix))
    result = [node]

    nodes_to_visit = list(range(len(dist_matrix)))
    nodes_to_visit.remove(node)

    while nodes_to_visit:
        nearest_node = min([(dist_matrix[node][j], j) for j in nodes_to_visit], key=lambda x: x[0])
        node = nearest_node[1]
        nodes_to_visit.remove(node)
        result.append(node)

    return result


class SimulatedAnnealing:
    def __init__(self, coords, temp, alpha, stopping_temp, stopping_iter):
        """
        Initial method for Simulated Annealing optimization
        :param coords: coordinates of cities
        :param temp: starting temperature of annealing process
        :param alpha: annealing coefficient
        :param stopping_temp: lower bound for a temperature
        :param stopping_iter: number of iterations
        """

        self.coords = coords
        self.sample_size = len(coords)
        self.temp = temp
        self.alpha = alpha
        self.stopping_temp = stopping_temp
        self.stopping_iter = stopping_iter
        self.iteration = 1

        self.dist_matrix = to_distance_matrix(coords)
        self.curr_solution = find_nearest_neighbours(self.dist_matrix)
        self.best_solution = self.curr_solution

        self.solution_history = [self.curr_solution]

        self.curr_weight = self.weight(self.curr_solution)
        self.initial_weight = self.curr_weight
        self.min_weight = self.curr_weight

        self.weight_list = [self.curr_weight]

        print('Initial weight: ', self.curr_weight)

    def weight(self, sol):
        """
        Evaluating weight based on the proposed solution.
        :param sol: proposed solution
        :return: weight
        """
        return sum([self.dist_matrix[i, j] for i, j in zip(sol, sol[1:] + [sol[0]])])

    def acceptance_probability(self, candidate_weight):
        """
        Calculating the probability of accepting new weights.
        :param candidate_weight: new weights
        :return: probability value
        """
        return math.exp(-abs(candidate_weight - self.curr_weight) / self.temp)

    def accept(self, candidate):
        """
        Used for processing new weights and update attributes according to the decision.
        Accept with probability 1 if candidate solution is better than
        current solution, else accept with probability equal to the
        acceptance_probability()
        :param candidate: proposed weights
        """
        candidate_weight = self.weight(candidate)
        if candidate_weight < self.curr_weight:
            self.curr_weight = candidate_weight
            self.curr_solution = candidate
            if candidate_weight < self.min_weight:
                self.min_weight = candidate_weight
                self.best_solution = candidate.copy()

        else:
            if random.random() < self.acceptance_probability(candidate_weight):
                self.curr_weight = candidate_weight
                self.curr_solution = candidate

    def anneal(self):
        """
        Simulated Annealing algorithm realization
        """
        while self.temp >= self.stopping_temp and self.iteration < self.stopping_iter:
            candidate = list(self.curr_solution)
            l = random.randint(2, self.sample_size - 1)
            i = random.randint(0, self.sample_size - l)

            candidate[i: (i + l)] = reversed(candidate[i: (i + l)])

            self.accept(candidate)
            self.temp *= self.alpha
            self.iteration += 1
            self.weight_list.append(self.curr_weight)
            self.solution_history.append(self.curr_solution)

        print('Minimum weight: ', self.min_weight)
        print('Improvement: ',
              round((self.initial_weight - self.min_weight) / (self.initial_weight), 4) * 100, '%')

    def animate_solutions(self):
        """
        Used to animate solution of a problem
        """
        visualization.animate_problem(self.solution_history, self.coords)

    def plot_learning(self):
        """
        Used to plot results of a learning
        """
        plt.plot([i for i in range(len(self.weight_list))], self.weight_list)
        line_init = plt.axhline(y=self.initial_weight, color='r', linestyle='--')
        line_min = plt.axhline(y=self.min_weight, color='g', linestyle='--')
        plt.legend([line_init, line_min], ['Initial weight', 'Optimized weight'])
        plt.ylabel('Weight')
        plt.xlabel('Iteration')
        plt.show()