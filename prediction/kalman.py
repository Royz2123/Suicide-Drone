#!/usr/bin/env python
# coding: utf-8

from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from filterpy.common import Q_discrete_white_noise
import json


class KalmanPredictor(object):
    def __init__(self, deg=2):
        self.deg = deg
        self.dim = 3 * (deg + 1)
        self.inverse_factorials = [1 / np.math.factorial(i) for i in range(1, deg + 1)]
        self.f = KalmanFilter(dim_x=self.dim, dim_z=self.dim)
        self.f.H = np.eye(self.dim)
        # f.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=0.0001)
        self.f.P *= 0.0001
        self.f.R = 0.0001 * np.eye(self.dim)
        self.times = np.array([])
        self.states = np.array([[]])
        self.prev_t = 0
        self.is_first_update = True

    def update(self, t, state):
        if not self.states.shape[1]:
            full_state = np.array([state + [0] * (self.dim - len(state))])
            self.f.x = full_state.T
            self.times = np.array([t])
            self.states = full_state.T
        elif self.states.shape[1] == 1:
            dt = t - self.times[0]
            full_state = np.array([state + [(state[0] - self.states[0][0]) / dt,
                                            (state[1] - self.states[1][0]) / dt, 0, 0]])
            self.set_dt(dt)
            self.f.update(full_state.T)
            self.times = np.append(self.times, t)
            self.states = np.append(self.states, full_state.T, axis=1)
        else:
            dt = t - self.times[0]
            dx = (state[0] - self.states[0][-1]) / dt
            dy = (state[1] - self.states[1][-1]) / dt
            ddx = (dx - self.states[2][-1]) / dt
            ddy = (dy - self.states[3][-1]) / dt
            full_state = np.array([state + [dx, dy, ddx, ddy]])
            self.set_dt(dt)
            self.f.update(full_state.T)
            self.times = np.append(self.times, t)
            self.states = np.append(self.states, full_state.T, axis=1)

    def predict_forward(self, t, dt):
        self.set_dt(dt)
        N = int(t / dt)
        z0 = self.f.x
        out = []
        for _ in range(N):
            self.f.predict_steadystate()
            out.append(self.f.x)
            print(self.f.x)
        self.f.update_steadystate(z0)
        return out

    def update_realtime(self, t, x):
        print(x)
        x = np.array(x)
        dt = t - self.prev_t
        self.prev_t = t
        prev_x = np.array(self.f.x)
        self.set_dt(dt)
        self.f.predict()
        print(self.f.x)
        dx = (x - prev_x[:3]) / dt
        if self.is_first_update:
            ddx = [0] * 3
            self.is_first_update = False
        else:
            ddx = (dx - prev_x[3:6]) / dt
        self.f.update(np.concatenate((x, dx, ddx)))

    def generate(self, test=dict()):
        t, x, y = test['t'], test['x'], test['y']
        x_ind = {p: index for index, p in enumerate(x)}
        y_ind = {p: index for index, p in enumerate(y)}
        x, y = sorted(x, key=lambda p: t[x_ind[p]]), sorted(y, key=lambda p: t[y_ind[p]])
        t = sorted(t)
        # t, x, y = np.array(t), np.array(x), np.array(y)
        # dts = np.diff(t)
        # derivs = [x, y]
        # dx = x
        # dy = y
        # for i in range(self.deg):
        #     dx = (dx[1:] - dx[:-1]) / dts
        #     dy = (dy[1:] - dy[:-1]) / dts
        #     dx = np.concatenate((np.array([0]), dx))
        #     dy = np.concatenate((np.array([0]), dy))
        #     derivs.append(dx)
        #     derivs.append(dy)
        # # states = np.vstack(derivs)

        self.f.x = [x[0], y[0], 0, 0, 0, 0, 0, 0, 0]
        self.prev_t = t[0]
        print(self.f.x)
        print(self.prev_t)
        out_dt = 0.01
        out_N = 10
        out_states = []
        for i in range(1, len(t)):
            self.update_realtime(t[i], [x[i], y[i], 0])

        # self.set_dt(f, out_dt)
        # for i in range(out_N):
        #     f.predict()
        #     f.update(f.x)
        # #     out_states.append(f.x)
        # out_states = np.vstack(out_states).T
        print('Starting prediction')
        out = self.predict_forward(5, 0.1)
        # return {'x': list(out_states[0, :]), 'y': list(out_states[1, :]),
        #         't': list(np.arange(t[-1] + out_dt, t[-1] + (out_N + 1)*out_dt, out_dt))}
        return {'x': [x[0] for x in out], 'y': [x[1] for x in out],
                't': list(np.arange(t[-1] + out_dt, t[-1] + (out_N + 1) * out_dt, out_dt))}

    def set_dt(self, dt):
        dt_powers = [dt ** i for i in range(1, self.deg + 1)]
        dt2 = dt ** 2
        F = np.eye(self.dim)
        for i in range(self.deg):
            for j in range(self.deg - i):
                F[3 * i:3 * (i + 1), 3 * (j + 1) + 3 * i:3 * (j + 2) + 3 * i] = \
                    self.inverse_factorials[j] * dt_powers[j] * np.eye(3)
        self.f.F = F

    def set_uncertainty(self, f, sigma):
        f.P *= sigma

    def load(self):
        return json.load(open('tests.json'))

    def predict(self, input_file, output_file, pred_sec=0.5):
        tests = json.load(open(input_file))

        out = dict()

        for k, test in tests.items():
            out[k] = self.generate(test)
        json.dump(out, open(output_file, 'w'))


class Kalman1(KalmanPredictor):
    def _init_(self):
        super().__init__(1)


class Kalman2(KalmanPredictor):
    def _init_(self):
        super().__init__(2)


class Kalman3(KalmanPredictor):
    def _init_(self):
        super().__init__(3)
