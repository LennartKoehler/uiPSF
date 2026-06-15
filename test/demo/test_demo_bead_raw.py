import sys
sys.path.append("../..")
from psflearning.psflearninglib import PSFLearningLib
from psflearning import Reader
from psflearning.writer import DefaultWriter

from psflearning import io

main_data_dir = 'example_data_for_uiPSF'
output_dir = 'test_output'

param = io.param.load_params(psftype='zernike', sysfile='M2')

reader = Reader()
writer = DefaultWriter()


# -- SETUP --
param = io.param.load_params(psftype='zernike', sysfile='M2')

# --- overwrite some params ---
param.io.data_path = main_data_dir+'/1ch_40nm_bead'
param.io.output_path = param.io.data_path+'psfmodel'
param.io.output_path = output_dir
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
from psflearning.progress import TqdmProgressReporter
reporter = TqdmProgressReporter()
parameters, psf_model, dataobj, learning_result, forward_images, context = PSFLearningLib.learn(param, images, reporter=reporter)
# -- SAVE --
parameters.io.output_path = output_dir

resfile = writer.save_result(parameters, context.pupil_field, dataobj, learning_result, forward_images=forward_images, reporter=reporter)
