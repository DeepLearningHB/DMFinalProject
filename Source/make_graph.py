import matplotlib.pyplot as plt
import numpy as np
import os


training_cost_list = []
test_cost_list = []
tr_mean_list = []
test_mean_list = []

f = open("./model/saved_final/log.txt", 'r')
log_list = f.readlines()

for log in log_list:
    log = log.replace("\n", '')
    log = log.split(" ")
    training_cost_list.append(float(log[2]))
    test_cost_list.append(float(log[4]))
    tr_mean_list.append(float(log[3]))
    test_mean_list.append(float(log[5]))



plt.plot(list(range(len(training_cost_list))), tr_mean_list, color='green')
plt.plot(list(range(len(test_cost_list))), test_mean_list, color='red')
plt.title("Training-Test Mean max distance Curve")
plt.ylim(0, 0.5)
plt.show()

