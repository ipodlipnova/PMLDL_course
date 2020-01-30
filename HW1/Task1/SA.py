import math
import random
import numpy as np

from model import *

class SimulatedAnnealing:
    def __init__(self, temp, alpha, stopping_temp, stopping_iter):
        """
        Initial method for Simulated Annealing optimization
        :param temp: starting temperature of annealing process
        :param alpha: annealing coefficient
        :param stopping_temp: lower bound for a temperature
        :param stopping_iter: number of iterations
        """
        self.temp = temp
        self.alpha = alpha
        self.stopping_temp = stopping_temp
        self.stopping_iter = stopping_iter
        self.iteration = 1
        self.best_acc = 0.0

        self.curr_solution = model.get_weights().copy()
        self.best_solution = self.curr_solution.copy()

        self.solution_history = [self.curr_solution.copy()]

        self.curr_weight = self.evaluation(self.curr_solution.copy())[0]
        self.initial_weight = self.curr_weight
        self.min_weight = self.curr_weight

        self.weight_list = [self.curr_weight]

        print('Initial loss: ', self.curr_weight)

    def evaluation(self, sol):
        """
        Evaluating the model with proposed weighs.  Tuple (loss, accuracy) is returned.
        :param sol: proposed solution
        :return: (loss, accuracy)
        """
        model.set_weights(sol)
        return model.evaluate(train_x, train_y)

    def random_neighbour(self, x):
        """
        Add some small random delta from uniform distribution to weights of a model.
        :param x: given weights
        :return: modified weights of the same shape as initial ones
        """
        for idx in range(len(x)):
            x[idx] += np.random.normal(0, 0.01, size=x[idx].shape)
        return x

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
        loss, acc = self.evaluation(candidate)
        if loss < self.curr_weight:
            self.curr_weight = loss
            self.curr_solution = candidate.copy()
            if loss < self.min_weight:
                self.min_weight = loss
                self.best_solution = candidate.copy()
                self.best_acc = acc

        elif random.random() < self.acceptance_probability(loss):
            self.curr_weight = loss
            self.curr_solution = candidate.copy()
        else:
            loss = self.evaluation(self.curr_solution.copy())[0]
            self.curr_weight = loss

    def anneal(self):
        """
        Simulated Annealing algorithm realization
        """
        while self.temp >= self.stopping_temp and self.iteration < self.stopping_iter:

            candidate = self.curr_solution.copy()
            candidate = self.random_neighbour(candidate.copy())

            self.accept(candidate.copy())
            self.temp *= self.alpha
            self.iteration += 1
            self.weight_list.append(self.curr_weight)
            self.solution_history.append(self.curr_solution.copy())

        print('Minimum weight: ', self.min_weight)
        print('Improvement: ',
              round((self.initial_weight - self.min_weight) / self.initial_weight, 4) * 100, '%')

