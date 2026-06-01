import sys
sys.path.append("../..")
from psflearning.psflearninglib import PSFLearningLib
from psflearning import io
import tensorflow as tf

main_data_dir = io.param.load('datapath.yaml').main_data_dir

try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('Running on GPU')
except Exception:
    print('Running on CPU')

resfile = main_data_dir + '/results/1ch_FD_bead/psfmodel_zernike_vector_FD_single.h5'
f, p = io.h5.load(resfile)
L = PSFLearningLib()

figs = L.plotter.plot_zernike_map(f, p)
if isinstance(figs, list):
    for f_fig in figs:
        f_fig.show()
else:
    figs.show()

strehlratio = L.calstrehlratio(p, f)

fig = L.plotter.plot_psf_vs_data(f, p, index=50)
fig.show()

fwhmx, fwhmy, fwhmz = L.calfwhm(p, f)
