
import pickle

from matplotlib import pyplot as plt

file_name = 'result_of_k_3_12_trial_1.pickle'

with open(file_name, 'rb') as f:
      recv_ratios = pickle.load(f)

#recv_ratios = recv_ratios[:-3]

plt.xticks(range(0, 12), range(3, 15))
plt.xlabel('Number of non-zero elements of true sparse signal', fontsize=20)

plt.ylabel('Recovery ratio [%]', fontsize=20)
plt.plot(recv_ratios, marker='x')
plt.show()
