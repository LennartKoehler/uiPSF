import sys
sys.path.append("../..")
from psflearning.psflearninglib import PSFLearningLib
from psflearning import io
import tensorflow as tf

main_data_dir = io.param.load('datapath.yaml').main_data_dir

try:
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
    print('Running on GPU')
except Exception:
    print('Running on CPU')

L = PSFLearningLib()
param = io.param.combine(basefile='config_base', psftype='insitu', channeltype='4pi', sysfile='4pi')

param.datapath = main_data_dir+'/4pi_smlm_nup96/'
param.savename = param.datapath + 'psfmodel_iter'
param.keyword = 'cell3'
param.gain = 0.44
param.ccd_offset = 100
param.roi.peak_height = 0.4
param.batch_size = 500
param.option.insitu.stage_pos = 0.55
param.ref_channel = 0
param.option.model.const_pupilmag = True
param.option.insitu.var_stagepos = False

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

fig = L.plotter.plot_transform(f)
fig.show()
import numpy as np
np.set_printoptions(precision=4, suppress=True)
print(f.res.T)

fig = L.plotter.plot_learned_params_insitu(f, p)
fig.show()

fig = L.plotter.plot_coordinates(f, p)
fig.show()

print('f:\n    ', list(f.keys()))
print(' locres:\n    ', list(f.locres.keys()))
print(' res:\n    ', list(f.res.keys()))
print(' rois:\n    ', list(f.rois.keys()))
