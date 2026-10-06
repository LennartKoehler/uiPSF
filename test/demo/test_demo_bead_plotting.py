import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append("../..")
from psflearning.psflearninglib import PSFLearningLib
from psflearning import Reader, Writer
from psflearning.writer import H5Writer

from psflearning import io
from psflearning import Plotter

import logging
logging.basicConfig(filename='test_output/uiPSF.log', level=logging.DEBUG, filemode='w')
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.colorbar').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
logging.getLogger('h5py').setLevel(logging.WARNING)


param = io.param.load_params(userfile='config_multi_bead_test',sysfile=None)

reader = Reader()
writer = H5Writer()




images = reader.read_images(param)
# -- RUN --
from psflearning.progress import TqdmProgressReporter
reporter = TqdmProgressReporter()
psf_model, dataobj, learning_result, forward_images, context, locres = PSFLearningLib.run(param, images, reporter=reporter)
# -- SAVE --

resfile = writer.save_result(param, context.pupil_field, dataobj, learning_result, forward_images, reporter=reporter)

# -- PLOT & SAVE --
print('\nGenerating plots and saving to:', param.io.output_path)
plotter: Plotter = Plotter()

saved = plotter.generate_report(learning_result, dataobj, forward_images, context.pupil_field, locres, param, param.io.output_path, max_index=forward_images.shape[0])
