from desat_rgb_noise import desat_rgb_noise
from improvement_methods import *


noisy_folder_prefix = '/home/ben/image_enhancement/experiment_images/noisy'
inc_sat_folder = '/home/ben/image_enhancement/experiment_images/inc_sat/'
larsson_folder = '/home/ben/image_enhancement/experiment_images/colornet/'
hard_hue_folder = '/home/ben/image_enhancement/experiment_images/denoise_net/'
originals_folder = '/home/ben/image_enhancement/experiment_images/classics/'
blur_saturation_folder = '/home/ben/image_enhancement/experiment_images/blur_hue_saturation/'
CBM3D_folder = '/home/ben/image_enhancement/experiment_images/CBM3D/'

method_folders = [inc_sat_folder, larsson_folder, hard_hue_folder, blur_saturation_folder, CBM3D_folder]

# Construct images by reducing saturation and adding noise to the images in a folder, and save it to new folders
# ==============================================================================================================
saturation_factors = [1, 0.5]
noise_levels = [5, 10]
green_factor = 0

desat_rgb_noise(originals_folder, noisy_folder_prefix, saturation_factors, noise_levels, green_factor=green_factor)
print "Reduced saturation and added noise"

noise_types = []
for s in saturation_factors:
    for n in noise_levels:
        noise_types.append("{}_{}_{}/".format(int(s*100), n, green_factor))

# Make the folder hierarchy
for method_folder in method_folders:
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)
        print "Made " + method_folder
    for noise_type in noise_types:
        if not os.path.exists(method_folder + noise_type):
            os.makedirs(method_folder + noise_type)
            print "Made " + method_folder + noise_type

# Improve the noisy images with different methods
# ==============================================================================================================
noisy_folders = []
for s in saturation_factors:
    for n in noise_levels:
        noisy_folders.append("{}_{}_{}_{}/".format(noisy_folder_prefix, int(s*100), n, green_factor))

# No networks
increase_sat(from_folders=noisy_folders, to_folder=inc_sat_folder, plotstuff=False)
print "Increased saturation"

blur_hue_saturation(from_folders=noisy_folders, to_folder=blur_saturation_folder, plotstuff=False)
print "Took blur and saturation from blurred input"

# Larsson's
cnn_noise_remover(from_folders=noisy_folders, to_folder=larsson_folder, hard_hue=False, plotstuff=False)
print "Recolored with Larssons CNN"

# My variant, hard hue
cnn_noise_remover(from_folders=noisy_folders, to_folder=hard_hue_folder, hard_hue=True, plotstuff=False)
print "Recolored with my method"


