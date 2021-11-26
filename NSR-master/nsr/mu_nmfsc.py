def _multiplicative_update(data, n_components, sc_dict=None, sc_code=None, max_iter=1000, tol=1e-4, true_dict=None, verbose=0, eval_log=None):

    """
    initialization
    """
    data = data / np.max(data)
    n_features, n_samples = data.shape
    dictionary = np.random.rand(n_features, n_components)
    code = np.random.rand(n_components, n_samples)

    data = data / np.max(data)
    code = code / (np.sqrt(sum(code ** 2)) * np.ones((1, n_samples)))


    if sc_dict is not None:
        L1_dict = n_features ** 0.5 - (n_features ** 0.5 - 1) * sc_dict
        for i in range(0, n_components):
            tmp_dict, _ = _projfunc(dictionary[:, i][:, np.newaxis], L1_dict, 1, nn=True)
            dictionary[:, i] = tmp_dict.reshape(-1)
    else:
        L1_dict = None

    if sc_code is not None:
        L1_code = n_samples ** 0.5 - (n_samples ** 0.5 - 1) * sc_code
        for i in range(0, n_components):
            tmp_code, _ = _projfunc(code[i, :][:, np.newaxis], L1_code, 1, nn=True)
            code[i, :] = tmp_code.reshape(-1)
    else:
        L1_code = None

    # initialize evaluation log

    if eval_log is True and true_dict is not None:
        time_lps, error, sparsity, atoms = \
            _evaluations(data, dictionary, code, true_dict=true_dict)
    else:
        time_lps, error, sparsity = \
            _evaluations(data, dictionary, code, true_dict=None)

    start_time = time_lps
    time_log = np.zeros(max_iter)
    error_log = np.zeros(max_iter)
    sparse_log = np.zeros(max_iter)

    if eval_log is True and true_dict is not None:
        atom_at_init = atoms
        previous_atom = atom_at_init

        atom_log = np.zeros(max_iter)

    objhistory = 0.5 * sum(sum((data - np.dot(dictionary, code)) ** 2))
    stepsize_dict = 1
    stepsize_code = 1
    for n_iter in range(1, max_iter + 1):

        if sc_code is not None:
            delta_code = np.dot(dictionary.T, (np.dot(dictionary, code) - data))
            begobj = objhistory

            while(True):
                code_new = code - stepsize_code * delta_code
                for i in range(0, n_components):
                    tmp_code, _ = _projfunc(code[i, :][:, np.newaxis], L1_code, 1, nn=True)
                    code_new[i, :] = tmp_code.reshape(-1)

                newobj = 0.5 * sum(sum((data - np.dot(dictionary, code_new)) ** 2))
                if newobj <= begobj:
                    break

                stepsize_code /= 2
                if stepsize_code < 1e-200:
                    print('Algorithm converged')
                    return dictionary, code, n_iter, logs

            stepsize_code *= 1.2
            code = code_new
        else:
            code = code * (np.dot(dictionary.T, data)) / \
            (np.dot(np.dot(dictionary.T, dictionary), code) + EPSILON)

            norms = np.sqrt(sum(dictionary.T ** 2))
            code /= (np.dot(norms[:, np.newaxis], np.ones((1, n_samples))))
            dictionary *= np.dot(np.ones((n_features, 1)), norms[np.newaxis, :])

        if sc_dict is not None:
            delta_dict = np.dot((np.dot(dictionary, code) - data), code.T)
            begobj = 0.5 * sum(sum((data - np.dot(dictionary, code)) ** 2))

            while(True):
                dict_new = dictionary - stepsize_dict * delta_dict
                norms = np.sqrt(sum(dict_new ** 2))
                for i in range(0, n_components):
                    tmp_dict, _ = _projfunc(new_dict[:, i][:, np.newaxis], L1_dict * norms[i],
                                            norms[i] ** 2, nn=True,
                                            verbose=verbose)
                    dict_new[:, i] = tmp_dict.reshape(-1)

                newobj = 0.5 * sum(sum((data - np.dot(dict_new, code)) ** 2))

                if newobj <= begobj:
                    break
                stepsize_dict /= 2

                if stepsize_dict < 1e-200:
                    print('Algorithm converged')
                    return dictionary, code, n_iter, logs

            stepsize_dict *= 1.2
            dictionary = dict_new

        else:
            dictionary = dictionary * (np.dot(data, code.T)) / \
            (np.dot(dictionary, np.dot(code, code.T)) + EPSILON)

        code_ = code.copy()
        code_[code_ < 0.0000001] = 0
        
        if eval_log:
            if true_dict is not None:
                time_lps, error, sparsity, atoms = _evaluations(data, dictionary, code, true_dict=true_dict)
            else: 
                time_lps, error, sparsity = _evaluations(data, dictionary, code, true_dict=None)


            logged_error = error
            logged_sp = sparsity
            iter_time = time_lps
            logged_time = iter_time - start_time

            error_log[n_iter - 1] = logged_error
            sparse_log[n_iter - 1] = np.mean(logged_sp)
            time_log[n_iter - 1] = logged_time

            if true_dict is not None:
                logged_atom = atoms
                atom_log[n_iter - 1] = logged_atom

        if n_iter % 10 == 0:

            error = np.linalg.norm(data - np.dot(dictionary, code), ord='fro')
            sp = _hoyers_sparsity(normalize(code, axis=1))

            if verbose:
                iter_time = time.time()

                if true_dict is not None and eval_log:
                    print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f, atom: %f" %
                          (n_iter, iter_time - start_time, error, np.mean(logged_sp), logged_atom))
                else:
                    print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f" %
                          (n_iter, iter_time - start_time, error, np.mean(sp)))

    if eval_log is None:
        return dictionary, code, n_iter

    else:
        if eval_log is True and true_dict is not None:
            logs = {'time': time_log, 'error': error_log,
                    'sparsity': sparse_log, 'atoms': atom_log}
        else:
            logs = {'time': time_log, 'error': error_log,
                    'sparsity': sparse_log}

        return dictionary, code, n_iter, logs
