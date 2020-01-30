import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, MovieWriter

from dataset_processing import coord, res_cities


def animate_problem(history, points):
    """
    Used to animate problem solution
    :param history: list of historical solutions
    :param points: coordinates of cities
    """

    key_frames_mult = len(history) // 1500

    fig, ax = plt.subplots()

    line, = plt.plot([], [], lw=2)

    def init():
        x = [points[i][0] for i in history[0]]
        y = [points[i][1] for i in history[0]]
        plt.plot(x, y, 'co', label='oi')

        for i in range(len(x)):
            ax.annotate(str(res_cities.loc[(res_cities['geo_lat'] == x[i]) & (res_cities['geo_lon'] == y[i])]['city'].values[0]), (x[i], y[i]))

        extra_x = (max(x) - min(x)) * 0.05
        extra_y = (max(y) - min(y)) * 0.05
        ax.set_xlim(max(x) + extra_x, min(x) - extra_x)
        ax.set_ylim(max(y) + extra_y, min(y) - extra_y)

        line.set_data([], [])
        return line,

    def update(frame):
        x = [points[i, 0] for i in history[frame] + [history[frame][0]]]
        y = [points[i, 1] for i in history[frame] + [history[frame][0]]]
        line.set_data(x, y)
        return line

    ani = FuncAnimation(fig, update, frames=range(0, len(history), key_frames_mult),
                        init_func=init, interval=3, repeat=False)
    ani.save(filename='sol.mp4')
    plt.show()

