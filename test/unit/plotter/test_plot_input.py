import matplotlib as plt
plt.use('Agg')

import sys
sys.path.append("../..")
from psflearning import io
from psflearning.plotter import Plotter, save_figs
from psflearning.reader import Reader
from psflearning.writer import DefaultWriter
from psflearning import PSFLearningLib

from psflearning.psf_registry import get_psf_info

main_data_dir = 'example_data_for_uiPSF'
output_dir = 'test_output'

reader = Reader()
writer = DefaultWriter()
plotter = Plotter()



# -- SETUP --
param = io.param.load_params(psftype='zernike', sysfile='M2')

param.io.datapath = main_data_dir+'/1ch_40nm_bead/'
param.io.savename = param.io.datapath+'psfmodel'
param.io.keyword = 'Pos'
param.io.subfolder = 'Pos'
param.data.gain = 0.22
param.data.ccd_offset = 400
param.selection.FOV.z_step = 1
param.data.pixel_size.z = 0.05
param.selection.roi.max_bead_number = 20
param.selection.roi.bead_radius = 0.025
param.runtime.batch_size = 30
param.data.emission_wavelength = 0.6
param.runtime.relearn = True


images = reader.read_images(param)

psf_info = get_psf_info(param)
dataobj = PSFLearningLib._prep_data(param, images)

index: int = 1
fig1 = plotter.plot_psf(dataobj.measured_roi_images[index], param.data.pixel_size.z)
save_figs(fig1, output_dir, "input_psf", fmt="png", dpi=150)

print("test_plot_input complete")
