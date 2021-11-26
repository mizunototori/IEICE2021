'''
ベクトルのスパースコーディングのデモ
y = Ax
y: 既知の観測信号 (n_feature, 1)
A: 既知の辞書 (n_feature, n_components)
x: 未知のスパースベクトル (n_components, 1)


目的:
- スパースコーディング単体で正しく動作しているかを確認する
'''


import numpy as np
from nsc import NSC
from nsr import _make_nn_sparse_coded_signal
from sklearn.preprocessing import normalize
from nsr import _count_atoms
from nsr import _count_codes
from nsr import _support
from nsr import _reorder_matrix
from nsr import _l2_error
from nsr import _support_dist
from nsr_utils import _get_psnr
from matplotlib import pyplot as plt
import pickle


def lp_exam():
    '''
    n_features = 5 # 20
    n_components = 10 # 50
    n_samples = 150 # 1500
    n_nonzero_coefs = 1
    '''
    n_features = 340
    n_components = 1024
    n_samples = 1
    n_nonzero_coefs = 5#160
    nn_method = 'rect'
    true_data, true_dictionary, true_code = \
        _make_nn_sparse_coded_signal(n_samples=n_samples,
                                     n_components=n_components,
                                     n_features=n_features,
                                     n_nonzero_coefs=n_nonzero_coefs,
                                     random_state=0,
                                     alpha=1)

    alpha = 0.01
    p = 0.99
    constraint = 'wl1'
    model = NSC(solver='mu',
                constraint=constraint, p=p, e=0.0001,
                alpha=alpha, max_iter=10000, tol=1e-10, verbose=1, eval_log=True)

    est_code, n_iter = model.fit_transform(true_data, true_dictionary, true_code, return_iter=True)
    est_logs = model.logs



    (markers, stemlines, baseline) = plt.stem(true_code)
    plt.setp(markers, markersize=4, label='Ground truth')
    (markers, stemlines, baseline) = plt.stem(est_code)
    plt.setp(markers, color='orange', marker='x', markersize=4, label='Estimated')
    plt.legend()
    plt.savefig('sc_coef_stem.png')
    plt.clf()

    plt.plot(true_code[np.nonzero(true_code)], label='Ground truth')
    plt.plot(est_code[np.nonzero(est_code)], label='Estimated')
    plt.legend()
    plt.savefig('sc_coef_plot.png')
    plt.clf()

    plt.plot(est_logs['psnr'][:n_iter])
    plt.savefig('sc_coef_psnr.png')
    plt.clf()
    
    sd = _support_dist(true_code, est_code)

    l2error =  np.mean([_l2_error(t, e) for t, e in zip(true_code, est_code)])
    code_ratio = _count_codes(true_code, est_code)
    psnr = _get_psnr(true_code, est_code)


    print("distance between supports: %r" % sd)
    print("l2 error: %r" % l2error)
    print("code recovery rate: %r" % code_ratio)
    print("PSNR: %r" % psnr)


    file_name = constraint + '_alpha_' + str(alpha) + '_p_' + str(p)

    save_objects = {'true_dictionary': true_dictionary,
                    'true_code': true_code,
                    'est_code': est_code,
                    'est_logs': est_logs}

    with open(file_name + '.pickle', 'wb') as f:
        pickle.dump(save_objects, f)

if __name__ == '__main__':
    lp_exam()
