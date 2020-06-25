#!/usr/bin/env python
# coding: utf-8
import json
import matplotlib.pyplot as plt
import numpy as np

from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

class KalmanPredictor(object):
    def __init__(self, x, deg=1, t=0):
        '''
        Kalman predictor constructor.
        @param x - Starting position vector (3 elements)
        @param deg - Degree of prediction. Can be 0, 1, or 2. Defaults to 1.
        @param t - Time reference point. Defaults to 0. Set this to something else if you want 
        the predictor to start counting time from something else.
        '''
        self.deg_lim = deg
        self.deg = 2
        self.dim = 3 * (self.deg + 1)
        self.inverse_factorials = [1 / np.math.factorial(i) for i in range(1, self.deg + 1)]

        self.f = KalmanFilter(dim_x=self.dim, dim_z=self.dim)
        self.f.H = np.eye(self.dim)
        self.f.x = np.array(x + 6*[0])
        
        self.f.P *= 0.0001
        self.f.R = 0.0001 * np.eye(self.dim)
        self.times = np.array([])
        self.states = np.array([[]])
        self.prev_t = t
        self.is_first_update = True

    def predict(self, t, dt):
        '''
        Predicts the trajectory for the next t seconds with a timestep of dt.
        @param t - The time interval to predict points in
        @param dt - The time step to divide the interval by
        @returns A list of t // dt points representing the predictions. Each point is a 3-element array (x,y,z).
        '''
        self.set_dt(dt)
        N = int(t / dt)
        z0 = self.f.x
        out = []
        for _ in range(N):
            self.f.predict_steadystate()
            out.append(self.f.x[0:3])
        self.f.update_steadystate(z0)
        return out

    def update(self, t, x):
        '''
        Feeds a new trajectory measurement point to the filter. Make sure to always call this with increasing values of t.
        @param t - The timestamp of the point
        @param x - A 3-element coordinate array representing the trajectory point.
        @returns None
        '''
        x = np.array(x)
        dt = t - self.prev_t
        self.prev_t = t
        prev_x = np.array(self.f.x)
        self.set_dt(dt)
        self.f.predict()
        dx = (x - prev_x[:3]) / dt
        if self.is_first_update:
            ddx = [0] * 3
            self.is_first_update = False
        else:
            ddx = (dx - prev_x[3:6]) / dt
        self.f.update(np.concatenate((x, dx, ddx)))

    def batch_predict(self, test=dict()):
        '''
        Used for testing purposes. Takes a data dict of the form {t: [...], x: [...], y: [...]}
        outputs a dict of form {x: [...], y: [...], t: [...]} representing predictions for
        10 timepoints after the ones in the input dictionary, spanning 0.5 seconds.
        '''

        # Data preprocessing
        t, x, y = test['t'], test['x'], test['y']
        x_ind = {p: index for index, p in enumerate(x)}
        y_ind = {p: index for index, p in enumerate(y)}
        x, y = sorted(x, key=lambda p: t[x_ind[p]]), sorted(y, key=lambda p: t[y_ind[p]])
        t = sorted(t)

        # Feeds all trajectory points into the kalmanfilter
        self.f.x = [x[0], y[0], 0, 0, 0, 0, 0, 0, 0]
        self.prev_t = t[0]
        out_dt = 0.05
        out_N = 10
        out_states = []
        for i in range(1, len(t)):
            self.update_realtime(t[i], [x[i], y[i], 0])

        # Predicts the next 10 points with dt=0.05
        print('Starting prediction')
        out = self.predict_forward(out_dt*out_N, out_dt)
        return {'x': [x[0] for x in out], 'y': [x[1] for x in out],
                't': list(np.arange(t[-1] + out_dt, t[-1] + (out_N + 1) * out_dt, out_dt))}

    def set_dt(self, dt):
        '''
        For internal use only. Sets the current timestep of the kalmanfilter.
        If you're calling this externally, you are probably doing something wrong.
        '''
        dt_powers = [dt ** i for i in range(1, self.deg + 1)]
        dt2 = dt ** 2
        # Dynamically computes the F-matrix
        F = np.eye(self.dim)
        for i in range(self.deg):
            for j in range(self.deg - i):
                F[3 * i:3 * (i + 1), 3 * (j + 1) + 3 * i:3 * (j + 2) + 3 * i] = \
                    self.inverse_factorials[j] * dt_powers[j] * np.eye(3)
        F[:, (self.deg_lim + 1)*3:] = 0
        self.f.F = F


    def set_uncertainty(self, f, sigma):
        '''
        Measurement uncertainty. Set this to optimize the filter.
        '''
        f.P *= sigma