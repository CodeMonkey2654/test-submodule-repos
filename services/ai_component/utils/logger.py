import os
import json
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.logs = {}

    def log(self, key, value):
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(value)

    def save_logs(self, filename='logs.json'):
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.logs, f)

    def plot_logs(self):
        for key, values in self.logs.items():
            plt.plot(values, label=key)
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
