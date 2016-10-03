from desat_rgb_noise import desat_rgb_noise
from evaluate_images import evaluate_images
from matplotlib import pyplot as plt
from improvement_methods import *
import autocolorize
import subprocess
import cv2

noisy_folder = '/home/ben/image_enhancement/experiment_images/noisy/'
inc_sat_folder = '/home/ben/image_enhancement/experiment_images/inc_sat/'
larsson_folder = '/home/ben/image_enhancement/experiment_images/larsson/'
hard_hue_folder = '/home/ben/image_enhancement/experiment_images/hard_hue/'
originals_folder = '/home/ben/image_enhancement/experiment_images/test/'
blur_chroma_folder = '/home/ben/image_enhancement/experiment_images/blur_hue_chroma/'

# Construct images by reducing saturation and adding noise to the images in a folder, and save it to another folder
# ==================================================================================================================
saturation_factors = [0.2, 0.8]
noise_levels = [20, 50]

desat_rgb_noise(originals_folder, noisy_folder, saturation_factors, noise_levels)
print "Reduced saturation and added noise"

# Improve the noisy images with different methods
# ==================================================================================================================
# No networks
increase_sat(from_folder=noisy_folder, to_folder=inc_sat_folder, plotstuff=False)
print "Increased saturation"
blur_hue_chroma(from_folder=noisy_folder, to_folder=blur_chroma_folder, plotstuff=False)
print "Took blur and chroma from blurred input"

# Larsson's
CNN_noise_remover(from_folder=noisy_folder, to_folder=larsson_folder, hard_hue=False, plotstuff=False)
print "Recolored with Larssons CNN"

# My variant, hard hue
CNN_noise_remover(from_folder=noisy_folder, to_folder=hard_hue_folder, hard_hue=True, plotstuff=True)
print "Recolored with my method"

# Compare the improved images with the original, non-noisy images
# ==================================================================================================================
evaluate_images(originals_folder, noisy_folder, 'Originals vs noisy.txt')
evaluate_images(originals_folder, inc_sat_folder, 'Originals vs inc_sat.txt')
evaluate_images(originals_folder, larsson_folder, 'Originals vs Larsson.txt')
evaluate_images(originals_folder, hard_hue_folder, 'Originals vs mine.txt')
evaluate_images(originals_folder, originals_folder,'Originals vs originals.txt') # psnr is inf, ssim is 1
evaluate_images(originals_folder, blur_chroma_folder,'Originals vs blur hue chroma.txt')

print "Evaluated the methods"
