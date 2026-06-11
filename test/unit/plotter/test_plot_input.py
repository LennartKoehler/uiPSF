import matplotlib as plt
plt.use('Agg')

import sys
sys.path.append("../..")
from psflearning import io
from psflearning.plotter import Plotter, save_figs
from psflearning.reader import Reader
from psflearning.writer import STDOUTWriter
from psflearning import PSFLearningLib

from psflearning.psf_registry import get_psf_info

main_data_dir = 'example_data_for_uiPSF'
output_dir = 'test_output'

reader = Reader()
writer = STDOUTWriter()
plotter = Plotter()

lib = PSFLearningLib(reader, writer)
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

# param.iterations = 250 # TESTVALUE

images = reader.read_images(param)
psf_info = get_psf_info(param)
dataobj = lib._prep_data(param, images)

index: int = 1
fig1 = plotter.plot_psf(dataobj.measured_roi_images[index], param.pixel_size.z)
save_figs(fig1, output_dir, "input_psf", fmt="png", dpi=150)

print("test_plot_input complete")
