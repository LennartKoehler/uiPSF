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
param = io.param.combine(basefile='config_base', psftype='insitu_FD', channeltype='1ch', sysfile='Ast_Li')

param.datapath = main_data_dir+'/1ch_FD_smlm_nup96/'
param.savename = param.datapath + 'psfmodel_2'
param.keyword = 'NUP'
param.subfolder = ''
param.gain = 0.47
param.ccd_offset = 100
param.pixel_size.z = 0.05
param.roi.peak_height = 0.2
param.roi.roi_size = [21, 21]
param.batch_size = 1500
param.option.model.const_pupilmag = True
param.option.insitu.stage_pos = 0.8
param.option.insitu.var_stagepos = False
param.loss_weight.smooth = 0.03
param.iteration = 200
param.insitu.min_photon = 0.2
param.option.insitu.partition_size = [11, 8, 8, 30]
param.option.insitu.repeat = 1
param.option.model.n_max = 6
param.option.model.division = 40

images = L.load_data(param)
psf_info = L.get_psf_info(param)
dataobj = L.prep_data(param, images)
resfile = L.iterlearn_psf(param, dataobj, time=0)

f, p = io.h5.load(resfile)

figs = L.plotter.plot_zernike_map(f, p)
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
