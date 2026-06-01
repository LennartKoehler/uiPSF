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

L = PSFLearningLib()
param = io.param.combine(basefile='config_base', psftype='insitu', channeltype='1ch', sysfile='TP')

param.datapath = main_data_dir+'/1ch_smlm_TP/'
param.savename = param.datapath+'psfmodel_iter'
param.keyword = 'data'
param.gain = 0.47
param.ccd_offset = 100
param.pixel_size.z = 0.1
param.roi.peak_height = 0.3
param.option.insitu.stage_pos = 2.4
param.batch_size = 1600
param.option.insitu.repeat = 6
param.option.insitu.min_photon = 0.7
param.option.model.blur_sigma = 0.6
param.option.model.const_pupilmag = True
param.plotall = True
param.PSFtype = 'insitu_pupil'

images = L.load_data(param)
psf_info = L.get_psf_info(param)
dataobj = L.prep_data(param, images)
resfile = L.iterlearn_psf(param, dataobj, time=0)

f, p = io.h5.load(resfile)

fig = L.plotter.plot_psf_vs_data_insitu(f, p)
if isinstance(fig, list):
    for f_fig in fig:
        f_fig.show()
else:
    fig.show()

figs = L.plotter.plot_pupil(f, p)
if isinstance(figs, list):
    for f_fig in figs:
        f_fig.show()
else:
    figs.show()

fig = L.plotter.plot_learned_params_insitu(f, p)
fig.show()

fig = L.plotter.plot_coordinates(f, p)
fig.show()

print('f:\n    ', list(f.keys()))
print(' locres:\n    ', list(f.locres.keys()))
print(' res:\n    ', list(f.res.keys()))
print(' rois:\n    ', list(f.rois.keys()))
