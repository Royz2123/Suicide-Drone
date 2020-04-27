import json
import numpy as np
import matplotlib.pyplot as plt


class PolyPredictor(object):
    OUTPUT_PREDS = 100
    POLY_DEG = 3

    def softmax_weights(self, n):
        x = np.linspace(0, n, num=n)
        return np.exp(x)/sum(np.exp(x))

    def poly_weights(self, n, d):
        x = np.linspace(0, n, num=n)
        print(x^2)
        return x^d/sum(x^d)

    def sigmoid_weights(self, n):
        x = np.linspace(0, n, num=n)
        return 1 / (1+np.exp(-x))

    def predict(self, input_file, output_file):
        with open(input_file, "r") as read_file:
            data = json.load(read_file)

        # set number of samples
        size = len(data)

        res = {}
        for i in range(len(data)):
            temp_data = data[str(i)]
            weights = self.softmax_weights(len(temp_data))

            t = temp_data['t']
            x = temp_data['x']
            y = temp_data['y']

            x_fit = np.polyfit(t, x, PolyPredictor.POLY_DEG, w=weights)
            poly_x = np.poly1d(x_fit)
            y_fit = np.polyfit(t, y, PolyPredictor.POLY_DEG, w=weights)
            poly_y = np.poly1d(y_fit)

            last_t = t[-1]
            t_arr = np.linspace(last_t, last_t + 0.5, num=size)
            x_t = poly_x(t_arr)
            y_t = poly_y(t_arr)

            new_dict = {}
            new_dict['t'] = t_arr.tolist()
            new_dict['x'] = x_t.tolist()
            new_dict['y'] = y_t.tolist()

            res[str(i)] = new_dict
            plt.plot(x_t, y_t, 'b')
            plt.plot(x, y, 'r')
            plt.title("plot number: "+str(i))
            plt.show()

        with open(output_file, "w") as write_file:
            json.dump(res, write_file)

