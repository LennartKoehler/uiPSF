import matplotlib as plt
plt.use('Agg')

import sys
sys.path.append("../..")
from psflearning.psflearninglib import PSFLearningLib
from psflearning import io
from psflearning.plotter import save_figs

main_data_dir = 'example_data_for_uiPSF'
output_dir = 'test_output'
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

index: int = 0
fig1 = L.plotter.plot_psf(dataobj.measured_roi_images[index], param.pixel_size.z)
save_figs(fig1, output_dir, "input_psf", fmt="png", dpi=150)

