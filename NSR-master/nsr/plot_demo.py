from matplotlib import pyplot as plt
import pickle
import numpy as np

constraint = 'wl1'
alpha = 0.35
p = 0.9

file_name = constraint + '_alpha_' + str(alpha) + '_p_' + str(p)

with open(file_name + '.pickle', 'rb') as f:
    save_objects = pickle.load(f)

true_dictionary = save_objects['true_dictionary']
true_code = save_objects['true_code']
est_dictionary = save_objects['est_dictionary']
est_coee = save_objects['est_code']
est_logs = save_objects['est_logs']

max_iter = len(est_logs['time'])

if est_logs['time'][-1] == 0.0:
    num_iter = np.where(est_logs['time'] == 0)[0][0]
else:
    num_iter = max_iter


plt.xscale('log')
plt.yscale('log')
plt.xlabel('Iteration Number')
plt.ylabel('Cost function Values')
plt.plot(est_logs['cost'][:num_iter])
plt.xlim(0, 10 ** 3)
plt.savefig('figures/' + file_name + '_cost_iter.png')

plt.figure(12)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time [s]')
plt.ylabel('Cost function Values')
plt.plot(est_logs['time'][:num_iter], est_logs['cost'][:num_iter])
plt.xlim(0, 10 ** 2)
plt.savefig('figures/' + file_name + '_cost_time.png')

plt.figure(21)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Iteration Number')
plt.ylabel('Approximation Error')
plt.plot(est_logs['error'][:num_iter])
plt.xlim(0, 10 ** 3)
plt.savefig('figures/' + file_name + '_error_iter.png')

plt.figure(22)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time [s]')
plt.ylabel('Approximation Error')
plt.plot(est_logs['time'][:num_iter], est_logs['error'][:num_iter])
plt.xlim(0, 10 ** 2)
plt.savefig('figures/' + file_name + '_error_time.png')

plt.figure(31)
plt.xscale('log')
plt.plot(est_logs['atoms'][:num_iter])
plt.ylabel('Recovery ratios [%]')
plt.xlabel('Iteration Number')
plt.xlim(0, 10 ** 3)
plt.ylim(0, 105)
plt.savefig('figures/' + file_name + '_atoms_iter.png')

plt.figure(32)
plt.xscale('log')
plt.plot(est_logs['time'][:num_iter], est_logs['atoms'][:num_iter])
plt.ylabel('Recovery ratios [%]')
plt.xlabel('Time [s]')
plt.xlim(0, 10 ** 2)
plt.ylim(0, 105)
plt.savefig('figures/' + file_name + '_atoms_time.png')

plt.figure(41)
plt.xscale('log')
plt.plot(est_logs['sparsity'][:num_iter])
plt.xlim(0, 10 ** 3)
plt.ylim(0, 1)
plt.ylabel('Sparseness')
plt.xlabel('Iteration Number')
plt.savefig('figures/' + file_name + '_sp_iter.png')

plt.figure(42)
plt.xscale('log')
plt.plot(est_logs['time'][:num_iter], est_logs['sparsity'][:num_iter])
plt.xlim(0, 10 ** 2)
plt.ylim(0, 1)
plt.ylabel('Sparseness')
plt.xlabel('Time [s]')
plt.savefig('figures/' + file_name + '_sp_time.png')

'''
plt.figure(11, figsize=(10, 10))
print('len(true_atoms), len(est_atoms):', len(true_atoms), len(est_atoms))
for i in range(1, len(true_atoms) + 1):
    plt.subplot(25, 2, i)
    plt.plot(true_atoms[i - 1])
    plt.plot(est_atoms[i - 1])
plt.savefig('figures/' + file_name + '_compare_atoms.png')


plt.figure(12, figsize=(100, 10))
# print('len(true_atoms), len(est_atoms):', len(true_codes), len(perm_code))
for i in range(1, 50 + 1):
    plt.subplot(5, 10, i)
    plt.stem(norm_true_code[:, i - 1], 'b-.')
    plt.stem(norm_est_code[:, i - 1].T, 'r-.', markersize=1)
plt.savefig('figures/' + file_name + '_compare_codes.png')
'''
