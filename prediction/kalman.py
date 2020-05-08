#!/usr/bin/env python
# coding: utf-8

from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from filterpy.common import Q_discrete_white_noise
import json

class KalmanPredictor(object):
    def __init__(self, deg=3):
        self.deg = deg
        self.dim = 2*(deg+1)
        self.inverse_factorials = [1/np.math.factorial(i) for i in range(1, deg + 1)]

    def generate(self, test):
        f = KalmanFilter (dim_x=self.dim, dim_z=self.dim)
        f.H = np.eye(self.dim)
        #f.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=0.0001)
        f.P *= 0.0001
        f.R = 0.0001 * np.eye(self.dim)
        t, x, y = test['t'], test['x'], test['y']
        x_ind = {p: index for index, p in enumerate(x)}
        y_ind = {p: index for index, p in enumerate(y)}
        x, y = sorted(x, key=lambda p: t[x_ind[p]]), sorted(y, key=lambda p: t[y_ind[p]])
        t = sorted(t)
        t, x, y = np.array(t), np.array(x), np.array(y)
        dts = np.diff(t)
        derivs = [x, y]
        dx = x
        dy = y
        for i in range(self.deg):
            dx = (dx[1:] - dx[:-1]) / dts
            dy = (dy[1:] - dy[:-1]) / dts
            dx = np.concatenate((np.array([0]), dx))
            dy = np.concatenate((np.array([0]), dy))
            derivs.append(dx)
            derivs.append(dy)
        states = np.vstack(derivs)

        f.x = states[:, 0]

        out_dt = 0.01
        out_N = 50
        out_states = []
        for i in range(1, states.shape[1]):
            self.set_dt(f, dts[i-1])
            f.predict()
            f.update(states[:, i])

        self.set_dt(f, out_dt)
        for i in range(out_N):
            f.predict()
            f.update(f.x)
            out_states.append(f.x)
        out_states = np.vstack(out_states).T
        return {'x': list(out_states[0, :]), 'y': list(out_states[1, :]),
                't': list(np.arange(t[-1] + out_dt, t[-1] + (out_N + 1)*out_dt, out_dt))}

    def set_dt(self, f, dt):
        dt_powers = [dt ** i for i in range(1, self.deg + 1)]
        dt2 = dt ** 2
        F = np.eye(self.dim)
        for i in range(self.deg):
            for j in range(self.deg - i):
                F[2*i:2*(i+1),2*(j+1) + 2*i:2*(j+2) + 2*i] = \
                    self.inverse_factorials[j]*dt_powers[j]*np.eye(2)
        f.F = F

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






