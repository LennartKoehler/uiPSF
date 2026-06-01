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

f, psfobj = L.genpsf(p, f, Nz=31, xsz=21, stagepos=3.0)
fig = L.plotter.plot_psf(f, p)
if isinstance(fig, list):
    for f_fig in fig:
        f_fig.show()
else:
    fig.show()

coeff = L.generate_cspline(p, f.res, psfobj)
f.locres.coeff = coeff
filename = resfile[:-3]+'_IMM.h5'
L.write_h5(p, filename, f.res, f.locres, f.rois)
