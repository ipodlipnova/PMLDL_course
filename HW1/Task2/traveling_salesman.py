from simulated_annealing import SimulatedAnnealing
from dataset_processing import coord


def main():
    # set the simulated annealing algorithm params
    temp = 1000
    stopping_temp = 0.00000001
    alpha = 0.9995
    stopping_iter = 10000000

    nodes = coord
    print(nodes)
    sa = SimulatedAnnealing(nodes, temp, alpha, stopping_temp, stopping_iter)
    sa.anneal()

    sa.animate_solutions()

    sa.plot_learning()


if __name__ == "__main__":
    main()