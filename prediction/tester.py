import matplotlib
matplotlib.use("Qt4Agg")

import matplotlib.pyplot as plt
import random
import numpy as np
import json

import matplotlib.animation as animation

plt.style.use('dark_background')



class Tester(object):
    TEST_FILE = "./prediction/tests.json"
    ANSWER_FILE = "./prediction/answers.json"
    AREA_RADIUS = 1

    SAMPLES = 5
    SUB_CASES = 5

    (
        SECRET,
        PLOT,
        ANIMATE,
        SAVE,
    )=range(4)

    def __init__(self, mode=ANIMATE):
        self._coefficients = None
        self._mode = mode

    def generate_test_file(self, cases=5):
        tests = {}
        answers = {}
        for case_num in range(cases):
            tests[case_num], answers[case_num] = self.get_trajectory()

        json.dump(tests, open(Tester.TEST_FILE, "w"))
        json.dump(answers, open(Tester.ANSWER_FILE, "w"))

    def generate_coefficients(self, degree):
        roots = [random.uniform(-Tester.AREA_RADIUS, Tester.AREA_RADIUS) for _ in range(degree)]
        self._coefficients = np.poly(roots)[::-1]

    def generate_polynomial(self):
        return lambda x: sum([c * (x ** i) for i, c in enumerate(self._coefficients)])

    def secret_function(self, x):
        # Serious Voodoo crap
        return np.sin(self.generate_polynomial()(x) * 1.5)

    def get_trajectory(self):
        t = np.linspace(-Tester.AREA_RADIUS, Tester.AREA_RADIUS, 10000)
        st = np.random.uniform(-Tester.AREA_RADIUS * 0.8, Tester.AREA_RADIUS * 0.5, Tester.SAMPLES)

        self.generate_coefficients(4)
        x = self.secret_function(t)
        sx = self.secret_function(st)

        self.generate_coefficients(4)
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

    def animate(self, t, x, y, sx, sy):
        fig = plt.figure()
        ax = plt.axes(xlim=(-Tester.AREA_RADIUS, Tester.AREA_RADIUS), ylim=(-Tester.AREA_RADIUS, Tester.AREA_RADIUS))
        line, = ax.plot([], [], lw=6)
        plt.plot(x, y, '--', linewidth=1)
        plt.scatter(sx, sy)

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
