import matplotlib.pyplot as plt

def save_plot(plot_func, filepath, title, xlabel, ylabel, xticks=None, legend=True):
    plt.figure(figsize=(10, 5))
    plot_func()
    if xticks is not None:
        plt.xticks(xticks, fontsize=18)
    else:
        plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=22)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    if legend:
        plt.legend(fontsize=18, title_fontsize=18)
    plt.grid(visible=True)
    plt.tight_layout()
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight')
