from matplotlib import pyplot as plt
import pickle
from nsr import _count_atoms

figure_name = "wl1_over_complete_alpha_18e-1"

with open(figure_name + '.pickle', 'rb') as f:
    saved_objects = pickle.load(f)

true_dictionary = saved_objects['true_dictionary']
true_code = saved_objects['true_code']
est_dictionary = saved_objects['est_dictionary']
est_code = saved_objects['est_code']

# atom_ratio, true_atoms, est_atoms = _count_atoms(est_dictionary, true_dictionary, axis=0, return_mat=True)

atom_ratio, true_atoms, est_atoms = _count_atoms(est_code, true_code, axis=0, return_mat=True, norm=False)
print('atom_ratio:', atom_ratio)
plt.figure(5, figsize=(10, 10))
print('len(true_atoms), len(est_atoms):', len(true_atoms), len(est_atoms))
for i in range(1, len(true_atoms) + 1):
    plt.subplot(25, 2, i)
    print('idx:', i - 1)
    plt.plot(true_atoms[i - 1])
    plt.plot(est_atoms[i - 1])
#plt.savefig('figures/' + figure_name + '_compare_atoms.png')
plt.show()