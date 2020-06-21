import matplotlib

matplotlib.use("Qt4Agg")

import matplotlib.pyplot as plt
import random
import numpy as np
import json
import os
import matplotlib.animation as animation
import time

import prediction.kalman as kalman
import prediction.poly as poly

plt.style.use('dark_background')


class Tester(object):
    TEST_FILE = "./prediction/tests/%s/tests.json"
    ANSWER_FILE = "./prediction/tests/%s/answers.json"
    VIDEO_FILE = './prediction/tests/%s/trajectory.mp4'

    BASE_FOLDER = "./prediction/tests/%s/"
    PREDICTION_FOLDER = "./prediction/tests/%s/predictions/"

    TRAJ_SECONDS = 3
    SAMPLE_SECONDS = 1.5
    PRED_SECONDS = 1

    SAMPLES = 15
    CASES = 1

    DEFAULT_SINUSES = 10

    PREDICTORS = [
        kalman.Kalman2,
        # poly.polyPredictor
    ]

    (
        SECRET,
        PLOT,
        ANIMATE,
        SAVE,
    ) = range(4)

    def __init__(self, mode=ANIMATE, test_name="default_test"):
        self._coefficients = None
        self._mode = mode

        self._frequencies = None
        self._phases = None

        self._test_name = test_name
        self._base_folder = Tester.BASE_FOLDER % self._test_name
        self._pred_folder = Tester.PREDICTION_FOLDER % self._test_name
        self._test_file = Tester.TEST_FILE % self._test_name
        self._answer_file = Tester.ANSWER_FILE % self._test_name
        self._video_file = Tester.VIDEO_FILE % self._test_name

        try:
            os.mkdir(self._base_folder)
            os.mkdir(self._pred_folder)
        except Exception as e:
            print("Test already exists, destroying" + str(e))

    def test_all_predictors(self):
        self.generate_test_file()

        pred_files = []
        for index, predictor_class in enumerate(Tester.PREDICTORS):
            pred_obj = predictor_class()
            pred_file = "%spreciction_%s.json" % (self._pred_folder, str(index))
            pred_obj.predict(
                input_file=self._test_file,
                output_file=pred_file,
                pred_sec=Tester.PRED_SECONDS
            )
            pred_files.append(pred_file)

        self.check_prediction_folder()

    def generate_test_file(self, cases=CASES):
        tests = {}
        answers = {}
        for case_num in range(cases):
            tests[case_num], answers[case_num] = self.get_trajectory()

        json.dump(tests, open(self._test_file, "w"))
        json.dump(answers, open(self._answer_file, "w"))

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

    def check_prediction_folder(self):
        answers = json.load(open(self._answer_file, "r"))
        test = json.load(open(self._test_file, "r"))

        pred_fnames = [self._pred_folder + fname for fname in os.listdir(self._pred_folder)]
        preds = [json.load(open(pred_file, "r")) for pred_file in pred_fnames]

        for test_num in range(Tester.CASES):
            curr_data = [data[str(test_num)] for data in preds]

            for test_case in preds[0].keys():
                test_data = test[test_case]
                answer_data = answers[test_case]

                if self._mode >= Tester.PLOT:
                    plt.plot(answer_data["x"], answer_data["y"])
                    plt.scatter(test_data["x"], test_data["y"])

                    for data in curr_data:
                        plt.plot(
                            data["x"],
                            data["y"],
                            '-.', linewidth=1
                        )
                    plt.show()

                if self._mode >= Tester.ANIMATE:
                    self.animate(
                        answer_data["t"],
                        answer_data["x"], answer_data["y"],
                        test_data["x"], test_data["y"],
                        curr_data
                    )

    def check_prediction_file(self, prediction_file):
        guess = json.load(open(prediction_file, "r"))
        answers = json.load(open(self._answer_file, "r"))
        test = json.load(open(self._test_file, "r"))

        for test_case in guess.keys():
            guess_data = guess[test_case]
            test_data = test[test_case]
            answer_data = answers[test_case]

            if self._mode >= Tester.PLOT:
                plt.plot(answer_data["x"], answer_data["y"])
                plt.scatter(test_data["x"], test_data["y"])

                plt.plot(guess_data["x"], guess_data["y"], '-.', linewidth=3, label=test_case)
                plt.show()
                plt.legend()

            if self._mode >= Tester.ANIMATE:
                self.animate(
                    answer_data["t"],
                    answer_data["x"], answer_data["y"],
                    test_data["x"], test_data["y"],
                    [guess_data]
                )

    def get_trajectory(self, method='sin'):
        N = 10000
        speed = 10
        sigma_omega = np.pi / 8
        dt = Tester.TRAJ_SECONDS / N
        t = np.linspace(0, Tester.TRAJ_SECONDS, N)
        st = np.sort(np.random.uniform(0, Tester.SAMPLE_SECONDS, Tester.SAMPLES))
        if method == 'sin':
            self.generate_sinuses()
            x = self.secret_function(t)
            sx = self.secret_function(st)

            self.generate_sinuses()
            y = self.secret_function(t)
            sy = self.secret_function(st)
        elif method == 'randomwalk':
            x = np.zeros(N)
            theta = 0
            for i in range(len):
                pass

        answer = {"t": t.tolist(), "x": x.tolist(), "y": y.tolist()}
        test = {"t": st.tolist(), "x": sx.tolist(), "y": sy.tolist()}

        if self._mode >= Tester.PLOT:
            plt.plot(x, y)
            plt.scatter(sx, sy)
            plt.show()

        if self._mode >= Tester.ANIMATE:
            self.animate(t, x, y, sx, sy)

        return test, answer

    def animate(self, t, x, y, sx, sy, guesses=[]):
        fig = plt.figure()
        ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))
        line, = ax.plot([], [], lw=6)

        plt.plot(x, y, '--', linewidth=1)
        plt.scatter(sx, sy)

        for i, pred in enumerate(guesses):
            plt.plot(pred["x"], pred["y"], '-.', linewidth=2, label=Tester.PREDICTORS[i].__name__)
        plt.legend(prop={'size': 10})

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
            anim.save(self._video_file, writer=mywriter)
