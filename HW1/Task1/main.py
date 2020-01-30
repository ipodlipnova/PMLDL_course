from SA import SimulatedAnnealing
from model import *

if __name__ == "__main__":
    print(network_classification())

    temp = 1000
    stopping_temp = 0.01
    alpha = 0.95
    stopping_iter = 1000

    sa = SimulatedAnnealing(temp, alpha, stopping_temp, stopping_iter)
    sa.anneal()

    model.set_weights(sa.best_solution)
    print(f'Best accuracy{sa.best_acc}')
    print(f'Loss and accuracy on train set {model.evaluate(train_x, train_y)}')
    print(f'Loss and accuracy on test set {model.evaluate(test_x, test_y)}')