import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append("../..")
from psflearning.psflearninglib import PSFLearningLib
from psflearning import Reader, Writer
from psflearning.writer import H5Writer

from psflearning import io
from psflearning import Plotter

main_data_dir = 'example_data_for_uiPSF'
output_dir = 'test_output'

param = io.param.combine('config_base', psftype='zernike', sysfile='M2')

reader = Reader()
writer = H5Writer()



# -- SETUP --
param = io.param.combine('config_base', psftype='zernike', sysfile='M2')

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
param.relearn = True


images = reader.read_images(param)
# -- RUN --
parameters, psf_model, dataobj, learning_result, loc_result, forward_images, context = PSFLearningLib.run(param, images)
# -- SAVE --

resfile = writer.save_result(parameters, context.pupil_field, dataobj, learning_result, loc_result, forward_images)

f, p = io.h5.load(resfile)

# -- PLOT & SAVE --
print('\nGenerating plots and saving to:', output_dir)
plotter: Plotter = Plotter()

saved = plotter.generate_report(f.res, f.rois, f.locres, p, output_dir, index=1)
