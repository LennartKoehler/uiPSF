import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append("../..")
from psflearning.psflearninglib import PSFLearningLib
from psflearning import io
import tensorflow as tf

main_data_dir = 'example_data_for_uiPSF'
output_dir = 'test_output'

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

# param.iterations = 250 # TESTVALUE

images = L.read_images(param)
psf_info = L.get_psf_info(param)
dataobj = L.prep_data(param, images)

# -- RUN --
for k in range(0, 1):
    psf_model, learning_result, loc_result, forward_images, toc = L.learn_with_relearn(param, dataobj, psf_info, time=0)


# -- SAVE --
resfile = L.save_result(param, psf_model, dataobj, learning_result, loc_result, forward_images=forward_images)

f, p = io.h5.load(resfile)

# -- PLOT & SAVE --
print('\nGenerating plots and saving to:', output_dir)
saved = L.plotter.generate_report(f.res, f.rois, f.locres, p, output_dir, index=1)
for name, paths in saved.items():
    print(f'  {name}:')
    for path in paths:
        print(f'    {path}')

print('\nf:\n    ', list(f.keys()))
print(' locres:\n    ', list(f.locres.keys()))
print(' res:\n    ', list(f.res.keys()))
print(' rois:\n    ', list(f.rois.keys()))
