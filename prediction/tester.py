import matplotlib
matplotlib.use("Qt4Agg")

import matplotlib.pyplot as plt
import random
import numpy as np
import json

import matplotlib.animation as animation

plt.style.use('dark_background')

class Tester(object):
    TEST_FILE = "./prediction/test/tests.json"
    ANSWER_FILE = "./prediction/test/answers.json"
    GUESS_FILE = "./prediction/test/guess.json"

    TRAJ_SECONDS = 2
    SAMPLE_SECONDS = 1.5

    SAMPLES = 5
    CASES = 10

    DEFAULT_SINUSES = 10

    (
        SECRET,
        PLOT,
        ANIMATE,
        SAVE,
    )=range(4)

    def __init__(self, mode=ANIMATE):
        self._coefficients = None
        self._mode = mode

        self._frequencies = None
        self._phases = None

    def generate_test_file(self, cases=CASES):
        tests = {}
        answers = {}
        for case_num in range(cases):
            tests[case_num], answers[case_num] = self.get_trajectory()

        json.dump(tests, open(Tester.TEST_FILE, "w"))
        json.dump(answers, open(Tester.ANSWER_FILE, "w"))

    def generate_coefficients(self, degree):
        roots = [random.uniform(0, Tester.SAMPLE_SECONDS) for _ in range(degree)]
        self._coefficients = np.poly(roots)[::-1]

    def generate_polynomial(self):
        return lambda x: sum([c * (x ** i) for i, c in enumerate(self._coefficients)])

    def generate_sinuses(self):
        self._frequencies = [random.uniform(0, 3) for _ in range(Tester.DEFAULT_SINUSES)]
        self._phases = [random.uniform(0, 100) for _ in range(Tester.DEFAULT_SINUSES)]

    def secret_function(self, x):
        return sum(
            [
                np.sin(x * freq + phase) / Tester.DEFAULT_SINUSES
                for freq, phase in zip(self._frequencies, self._phases)
            ]
        )

    def secret_function_2(self, x):
        # Serious Voodoo crap, should do more sin shit
        return np.sin(self.generate_polynomial()(x) * 1.5)

    def check_test_file(self):
        guess = json.load(open(Tester.GUESS_FILE, "r"))
        answers = json.load(open(Tester.ANSWER_FILE, "r"))
        test = json.load(open(Tester.TEST_FILE, "r"))

        for test_case in guess.keys():
            guess_data = guess[test_case]
            test_data = test[test_case]
            answer_data = answers[test_case]

            if self._mode >= Tester.PLOT:
                plt.plot(answer_data["x"], answer_data["y"])
                plt.scatter(test_data["x"], test_data["y"])
                plt.plot(guess_data["x"], guess_data["y"], '-.', linewidth=1)
                plt.show()

            if self._mode >= Tester.ANIMATE:
                self.animate(
                    answer_data["t"],
                    answer_data["x"], answer_data["y"],
                    test_data["x"], test_data["y"],
                    guess_data["x"], guess_data["y"]
                )

    def get_trajectory(self):
        t = np.linspace(0, Tester.TRAJ_SECONDS, 10000)
        st = np.random.uniform(0, Tester.SAMPLE_SECONDS, Tester.SAMPLES)

        self.generate_sinuses()
        x = self.secret_function(t)
        sx = self.secret_function(st)

        self.generate_sinuses()
        y = self.secret_function(t)
        sy = self.secret_function(st)

        answer = {"t": t.tolist(), "x": x.tolist(), "y": y.tolist()}
        test = {"t": st.tolist(), "x": sx.tolist(), "y": sy.tolist()}

        if self._mode >= Tester.PLOT:
            plt.plot(x, y)
            plt.scatter(sx, sy)
            plt.show()

        if self._mode >= Tester.ANIMATE:
            self.animate(t, x, y, sx, sy)

        return test, answer

    def animate(self, t, x, y, sx, sy, sx_guess=[], sy_guess=[]):
        fig = plt.figure()
        ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))
        line, = ax.plot([], [], lw=6)

        plt.plot(x, y, '--', linewidth=1)
        plt.scatter(sx, sy)

        plt.plot(sx_guess, sy_guess, '-.', linewidth=1)

        # initialization function
        def init():
            # creating an empty plot/frame
            line.set_data([], [])
            return line,

        # lists to store x and y axis points
        xdata, ydata = [], []
        speed_up = 30

        # animation function
        def animate(i, x, y):
            # Speed up animation
            i *= speed_up

            # appending new points to x, y axes points list
            xdata.append(x[i])
            ydata.append(y[i])

            xdata[:] = xdata[-5:]
            ydata[:] = ydata[-5:]

            line.set_data(xdata, ydata)
            return line,

        # setting a title for the plot
        plt.title('Drone Trajectory Simulation')
        # hiding the axis details
        plt.axis('off')

        # call the animator
        anim = animation.FuncAnimation(fig, lambda i: animate(i, x, y), init_func=init,
                                       frames=len(t) // speed_up, interval=1, blit=True)
        plt.show()

        if self._mode >= Tester.SAVE:
            # save the animation as mp4 video file
            plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/t8637523/Desktop/Rare Programs/ffmpeg/bin/ffmpeg.exe'
            mywriter = animation.FFMpegWriter(fps=60)
            anim.save('./prediction/videos/trajectory.mp4', writer=mywriter)
