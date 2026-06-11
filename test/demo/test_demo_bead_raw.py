import sys
sys.path.append("../..")
from psflearning.psflearninglib import PSFLearningLib
from psflearning import Reader, STDOUTWriter
from psflearning.writer import H5Writer

from psflearning import io
from psflearning import Plotter

main_data_dir = 'example_data_for_uiPSF'
output_dir = 'test_output'

param = io.param.combine('config_base', psftype='zernike', sysfile='M2')

reader = Reader()
writer = STDOUTWriter()



# -- SETUP --
param = io.param.combine('config_base', psftype='zernike', sysfile='M2')

images = reader.read_images(param)
# -- RUN --
parameters, psf_model, dataobj, learning_result, loc_result, forward_images, context = PSFLearningLib.run(param, images)
# -- SAVE --

resfile = writer.save_result(parameters, context.pupil_field, dataobj, learning_result, loc_result, forward_images)
