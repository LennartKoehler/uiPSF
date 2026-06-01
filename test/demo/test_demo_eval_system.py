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

resfile = main_data_dir+'results/1ch_40nm_bead/psfmodel_zernike_vector_single.h5'
f, p = io.h5.load(resfile)
L = PSFLearningLib()

try:
    figs = L.plotter.plot_zernike(f, p)
    if isinstance(figs, list):
        for f_fig in figs:
            f_fig.show()
    else:
        figs.show()
except Exception:
    pass

strehlratio = L.calstrehlratio(p, f)

if 'insitu' in p.PSFtype:
    fig = L.plotter.plot_psf_vs_data_insitu(f, p)
else:
    fig = L.plotter.plot_psf_vs_data(f, p, index=0)
if isinstance(fig, list):
    for f_fig in fig:
        f_fig.show()
else:
    fig.show()

try:
    fwhmx, fwhmy, fwhmz = L.calfwhm(p, f)
except Exception:
    print('fwhm not found')
