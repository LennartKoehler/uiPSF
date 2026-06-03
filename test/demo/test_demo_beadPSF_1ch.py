import sys
sys.path.append("../..")
from psflearning.psflearninglib import PSFLearningLib
from psflearning import io
import tensorflow as tf

# main_data_dir = io.param.load('test/demo/datapath.yaml').main_data_dir
main_data_dir = 'example_data_for_uiPSF'

try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('Running on GPU')
except Exception:
    print('Running on CPU')

L = PSFLearningLib()


# -- SETUP --
param = io.param.combine('config_base', psftype='zernike', channeltype='1ch', sysfile='M2')

param.datapath = main_data_dir+'/1ch_40nm_bead/'
param.savename = param.datapath+'psfmodel'
param.keyword = 'Pos'
param.subfolder = 'Pos'
param.gain = 0.22
param.ccd_offset = 400
param.FOV.z_step = 1
param.pixel_size.z = 0.05
param.roi.max_bead_number = 20
param.roi.bead_radius = 0.025
param.batch_size = 30
param.option.imaging.emission_wavelength = 0.6

images = L.load_data(param)
psf_info = L.get_psf_info(param)
dataobj = L.prep_data(param, images)

# -- RUN --
for k in range(0, 1):
    psfobj, fitter, learning_result, loc_result = L.learn_psf(param, dataobj, psf_info, time=0)


# -- SAVE --
resfile = L.save_result(param, psfobj, dataobj, fitter, learning_result, loc_result)

f, p = io.h5.load(resfile)

fig = L.plotter.plot_psf_vs_data(f, p, index=1)
fig.show()

fig = L.plotter.plot_localization(f, p)
if isinstance(fig, list):
    for f_fig in fig:
        f_fig.show()
else:
    fig.show()

try:
    figs = L.plotter.plot_zernike(f, p)
    if isinstance(figs, list):
        for f_fig in figs:
            f_fig.show()
    else:
        figs.show()
except Exception:
    try:
        figs = L.plotter.plot_pupil(f, p)
        if isinstance(figs, list):
            for f_fig in figs:
                f_fig.show()
        else:
            figs.show()
    except Exception:
        print('no pupil')

fig = L.plotter.plot_learned_params(f, p)
fig.show()

fig = L.plotter.plot_coordinates(f, p)
fig.show()

print('f:\n    ', list(f.keys()))
print(' locres:\n    ', list(f.locres.keys()))
print(' res:\n    ', list(f.res.keys()))
print(' rois:\n    ', list(f.rois.keys()))
