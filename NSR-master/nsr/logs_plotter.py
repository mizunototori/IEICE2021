from matplotlib import pyplot as plt
def atom_iter_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces, ):
    plt.xlim(0, 3000)
    plt.ylim(0,102)
    plt.ylabel('Recovery ratios (%)')
    plt.xlabel('Iteration Number')
    atom_iter_plots = []
    for (i, model) in enumerate(models):
         atom_iter_plots.append(plt.plot(model.logs['atoms'][:max_idxs[i]], linestyle=linestyles[i], marker=markers[i], ms=10, linewidth=2, color=mcolors[i], markerfacecolor=mfaces[i], markevery=200))

    plt.legend(model_names)

def atom_time_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces, ):
    plt.xlim(0,200)
    plt.ylim(0,102)
    plt.ylabel('Recovery ratios (%)')
    plt.xlabel('Running Time (s)')
    atom_time_plots = []
    for (i, model) in enumerate(models):
         atom_time_plots.append(plt.plot(model.logs['time'][:max_idxs[i]], model.logs['atoms'][:max_idxs[i]], linestyle=linestyles[i], marker=markers[i], ms=10, linewidth=2, color=mcolors[i], markerfacecolor=mfaces[i], markevery=200))

    plt.legend(model_names)

def error_iter_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces, ):
    plt.xlim(0, 3000)
    #plt.ylim(0,102)
    plt.ylabel('Approximation Error')
    plt.xlabel('Iteration Number')
    atom_iter_plots = []
    for (i, model) in enumerate(models):
         atom_iter_plots.append(plt.plot(model.logs['error'][:max_idxs[i]], linestyle=linestyles[i], marker=markers[i], ms=10, linewidth=2, color=mcolors[i], markerfacecolor=mfaces[i], markevery=200))

    plt.legend(model_names)

def error_time_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces, ):
    plt.xlim(0,200)
    # plt.ylim(0,102)
    plt.ylabel('Approximation Error')
    plt.xlabel('Running Time (s)')
    atom_time_plots = []
    for (i, model) in enumerate(models):
         atom_time_plots.append(plt.plot(model.logs['time'][:max_idxs[i]], model.logs['error'][:max_idxs[i]], linestyle=linestyles[i], marker=markers[i], ms=10, linewidth=2, color=mcolors[i], markerfacecolor=mfaces[i], markevery=200))

    plt.legend(model_names)

def cost_iter_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces, ):
    plt.xlim(0, 2500)
    #plt.ylim(0,102)
    plt.ylabel('Cost')
    plt.xlabel('Iteration Number')
    atom_iter_plots = []
    for i in range(0, 2):
         atom_iter_plots.append(plt.plot(models[i].logs['cost'][:max_idxs[i]], linestyle=linestyles[i], marker=markers[i], ms=10, linewidth=2, color=mcolors[i], markerfacecolor=mfaces[i], markevery=200))

    plt.legend(model_names)

def cost_time_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces, ):

    plt.xlim(0,100)
    plt.ylabel('Approximation Error')
    plt.xlabel('Running Time (s)')
    atom_time_plots = []
    for i in range(0, 2):
         atom_time_plots.append(plt.plot(models[i].logs['time'][:max_idxs[i]], models[i].logs['cost'][:max_idxs[i]], linestyle=linestyles[i], marker=markers[i], ms=10, linewidth=2, color=mcolors[i], markerfacecolor=mfaces[i], markevery=200))

    plt.legend(model_names)

def sp_iter_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces, ):

    plt.xlim(0, 3000)
    plt.ylim(0,1)
    plt.ylabel('Sparseness')
    plt.xlabel('Iteration Number')
    atom_iter_plots = []
    for (i, model) in enumerate(models):
         atom_iter_plots.append(plt.plot(model.logs['sparsity'][:max_idxs[i]], linestyle=linestyles[i], marker=markers[i], ms=10, linewidth=2, color=mcolors[i], markerfacecolor=mfaces[i], markevery=200))

    plt.legend(model_names)

def sp_time_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces, ):
    plt.xlim(0,200)
    plt.ylim(0,1)
    plt.ylabel('Sparseness')
    plt.xlabel('Running Time (s)')
    atom_time_plots = []
    for (i, model) in enumerate(models):
         atom_time_plots.append(plt.plot(model.logs['time'][:max_idxs[i]], model.logs['sparsity'][:max_idxs[i]], linestyle=linestyles[i], marker=markers[i], ms=10, linewidth=2, color=mcolors[i], markerfacecolor=mfaces[i], markevery=200))

    plt.legend(model_names)

def plot_all(fig_dir, models, model_names, max_idxs, linestyles, markers, mcolors, mfaces, ):
    plt.clf()
    atom_iter_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces)
    plt.savefig(fig_dir + 'atom_iter.png')
    plt.clf()
    atom_time_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces)
    plt.savefig(fig_dir + 'atom_time.png')

    plt.clf()
    error_iter_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces)
    plt.savefig(fig_dir + 'error_iter.png')
    plt.clf()
    error_time_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces)
    plt.savefig(fig_dir + 'error_time.png')

    plt.clf()
    cost_iter_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces)
    plt.savefig(fig_dir + 'cost_iter.png')
    plt.clf()
    cost_time_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces)
    plt.savefig(fig_dir + 'cost_time.png')

    plt.clf()
    sp_iter_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces)
    plt.savefig(fig_dir + 'sp_iter.png')
    plt.clf()
    sp_time_plot(models, model_names, max_idxs, linestyles, markers, mcolors, mfaces)
    plt.savefig(fig_dir + 'sp_time.png')




