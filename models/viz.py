import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def make_coalescent_heatmap(model_name, averaged_data_tuple, dpi=200):
    # f = plt.figure(, dpi=dpi)
    f, ax = plt.subplots(1, 1, dpi=dpi)  # figsize=(200, 10)
    im0 = ax.imshow(averaged_data_tuple[0], cmap='Wistia', aspect='auto')
    ax.plot(averaged_data_tuple[1], lw=1, c='black')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax)
    
    plt.suptitle("Coalescent heatmap distribution by {} model".format(model_name), fontsize=20)
    ax.set_title("Softmax")
    plt.xlabel('site position')
    plt.ylabel('')
    ax.yaxis.set_ticks(np.arange(0, averaged_data_tuple[0].shape[0], step=1))
    return f