from racing_dataset.read_leica import parseTextFile
from kalman import KalmanPredictor
from matplotlib import pyplot as plt
import numpy as np

DATASETS = ["leica.txt"] + [f"leica ({n}).txt" for n in range(2,17)]

def generate_residuals(time, pos):
	predictions = [pos[:, 0]]
	k = KalmanPredictor(list(pos[:, 0]), deg=1)
	k.prev_time = time[0]
	predictions = []

	for i in range(1, len(time)):
		t = time[i]
		k.update(t, pos[:, i])
		if i < len(time) - 1:
			next_t = time[i+1]
			dt = next_t - t
			if dt == 0:
				pred = k.predict(0.1, 0.1)
			else:
				pred = k.predict(dt, dt)
			predictions.append(pred[-1])
	predictions = np.array([[state[0], state[1], state[2]] for state in predictions]).T
	res = np.abs(pos[:, 2:] - predictions)
	return res

def error_graph():
	time, pos = parseTextFile('racing_dataset/data/leica.txt')
	time = [t.to_sec() for t in time]

	times = []
	accs = []

	for ds in DATASETS:
		time, pos = parseTextFile('racing_dataset/data/' + ds)
		time = [t.to_sec() for t in time]
		res = generate_residuals(time, pos)
		for i in range(2, len(time)):
			for axis in range(3):
				times.append(time[i] - time[i-1])
				accs.append(res[axis, i - 2])

	b0 = np.min(times)
	b1 = np.max(times)
	bins = np.linspace(b0, b1, 10)

	indices = np.digitize(times, bins)
	grouped_accs = [[]]*10
	for i in range(len(accs)):
		grouped_accs[indices[i] - 1].append(accs[i])
	means = [np.mean([x for x in bucket if not np.isnan(x)]) for bucket in grouped_accs]
	plt.scatter(times, accs)
	plt.show()

def process_trajectory(leica_index):
	time, pos = parseTextFile('racing_dataset/data/' + DATASETS[leica_index])
	time = [t.to_sec() for t in time]
	predictions = [pos[:, 0]]
	k = KalmanPredictor(list(pos[:, 0]), deg=1, t=time[0])

	for i in range(1, len(time) // 2):
		t = time[i]
		k.update(t, pos[:, i])

	predictions = k.predict(5, 0.2)
	plt.scatter(pos[0, :2 * len(time) // 3], pos[1, :2 * len(time) // 3], label='Real data')
	plt.scatter([p[0] for p in predictions], [p[1] for p in predictions], label='Predictions')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	error_graph()