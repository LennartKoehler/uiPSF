import sys
import logging
sys.path.append("../..")
from psflearning.psflearninglib import PSFLearningLib
from psflearning import Reader
from psflearning.writer import DefaultWriter
from psflearning import io


param = io.param.load_params(psftype='zernike', sysfile='M2')

reader = Reader()
writer = DefaultWriter()


logging.basicConfig(filename='uiPSF.log', level=logging.DEBUG)
# -- SETUP --
param = io.param.load_params(userfile='config_user', psftype='zernike', sysfile='M2')



images = reader.read_images(param)
# -- RUN --
from psflearning.progress import TqdmProgressReporter
reporter = TqdmProgressReporter()
psf_model, dataobj, learning_result, forward_images, context = PSFLearningLib.learn(param, images, reporter=reporter)
# -- SAVE --

resfile = writer.save_result(param, context.pupil_field, dataobj, learning_result, forward_images=forward_images, reporter=reporter)
