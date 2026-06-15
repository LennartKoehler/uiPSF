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

param = io.param.load_params(psftype='zernike', sysfile='M2')

reader = Reader()
writer = H5Writer()



# -- SETUP --
param.io.data_path = main_data_dir+'/1ch_40nm_bead/'
param.io.output_path = param.io.data_path+'psfmodel'
param.io.keyword = 'Pos'
param.io.subfolder = 'Pos'
param.data.camera_gain = 0.22
param.data.camera_offset = 400
param.selection.FOV.z_step = 1
param.data.pixel_size.z = 0.05
param.selection.roi.max_bead_number = 20
param.selection.roi.bead_radius = 0.025
param.runtime.batch_size = 30
param.data.emission_wavelength = 0.6
param.runtime.enable_relearning = True


images = reader.read_images(param)
# -- RUN --
parameters, psf_model, dataobj, learning_result, loc_result, forward_images, context = PSFLearningLib.run_with_localization(param, images)
# -- SAVE --

resfile = writer.save_result(parameters, context.pupil_field, dataobj, learning_result, loc_result, forward_images)

f, p = io.h5.load(resfile)

# -- PLOT & SAVE --
print('\nGenerating plots and saving to:', output_dir)
plotter: Plotter = Plotter()

saved = plotter.generate_report(f.res, f.rois, f.locres, p, output_dir, index=1)
