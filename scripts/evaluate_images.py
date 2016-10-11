import cv2
import os
import re
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

def evaluate_images(noise_folder, method_folders, plotstuff=False):

    # Set up names and paths
    match_object = re.search(r'noisy_(\d*_\d*_\d*)/', noise_folder)
    noise_type = match_object.group(1)
    textfile_psnr = '/home/ben/image_enhancement/experiment_images/{}_psnr.txt'.format(noise_type)
    textfile_ssim = '/home/ben/image_enhancement/experiment_images/{}_ssim.txt'.format(noise_type)


    # Prepare the text files
    methods = []
    for method_folder in method_folders:
        match_object = re.search(r'experiment_images/(.*)/', method_folder)
        methods.append(match_object.group(1))

    with open(textfile_psnr, "w") as myfile_psnr, open(textfile_ssim, "w") as myfile_ssim:
        myfile_psnr.write("{:30}".format('Image name'))
        myfile_ssim.write("{:30}".format('Image name'))

    with open(textfile_psnr, "a") as myfile_psnr, open(textfile_ssim, "a") as myfile_ssim:
        method_averages = {}
        for method in methods:
            method_averages{method + "_psnr"} = 0  # Keep track of average score for each method
            method_averages{method + "_ssim"} = 0
            myfile_psnr.write("{:20}".format(method))
            myfile_ssim.write("{:20}".format(method))
            print method_averages

    # Evaluate one image and one noise setting for all methods before switching to the next image
    filenames = sorted(os.listdir(noise_folder))
    number_of_images = len(filenames)
    for filename in filenames:
        with open(textfile_psnr, "a") as myfile_psnr, open(textfile_ssim, "a") as myfile_ssim:
            myfile_psnr.write('\n{:30}'.format(filename[:-4]))  # Remove file ending
            myfile_ssim.write('\n{:30}'.format(filename[:-4]))

            for method_folder in method_folders:

                input_img = cv2.imread(noise_folder + filename)
                output_img = cv2.imread(method_folder + noise_type + '/' + filename)

                if input_img is None:
                    raise Exception("Image {} wasn't read".format(noise_folder + filename))
                if output_img is None:
                    raise Exception("Image {} wasn't read".format(method_folder + noise_type + '/' + filename))

                my_psnr = psnr(input_img, output_img)
                my_ssim = ssim(input_img.mean(-1), output_img.mean(-1), range=input_img.min() - input_img.max())

                myfile_psnr.write("{:.2f}{:15}".format(my_psnr, ''))
                myfile_ssim.write("{:.4f}{:14}".format(my_ssim, ''))

                method_averages{method + "_psnr"} += 1  # Sum the scores, average later
                method_averages{method + "_ssim"} += 1

                if plotstuff:
                    match_object = re.search(r'experiment_images/(.*)/', method_folder)
                    method = match_object.group(1)

                    plt.suptitle('Method: {}'.format(method))
                    plt.subplot(121), plt.title('Input')
                    plt.imshow(input_img[:, :, ::-1])
                    plt.subplot(122), plt.title('Output')
                    plt.imshow(output_img[:, :, ::-1])
                    plt.show()

    method_averages /= number_of_images

    with open(textfile_psnr, "a") as myfile_psnr, open(textfile_ssim, "a") as myfile_ssim:
        myfile_psnr.write('\n{:30}'.format("Average"))
        myfile_ssim.write('\n{:30}'.format("Average"))
            
        for method in methods:
            myfile_psnr.write('{:20}'.format(method_averages{method + "_psnr"}))
            myfile_ssim.write('{:20}'.format(method_averages{method + "_ssim"}))
                        

CBM3D_folder = '/home/ben/image_enhancement/experiment_images/CBM3D/'
inc_sat_folder = '/home/ben/image_enhancement/experiment_images/inc_sat/'
larsson_folder = '/home/ben/image_enhancement/experiment_images/colornet/'
hard_hue_folder = '/home/ben/image_enhancement/experiment_images/denoise_net/'
originals_folder = '/home/ben/image_enhancement/experiment_images/classics/'
blur_saturation_folder = '/home/ben/image_enhancement/experiment_images/blur_hue_saturation/'

method_folders = [CBM3D_folder, inc_sat_folder, larsson_folder, hard_hue_folder, blur_saturation_folder]

# Compare the improved images with the original, non-noisy images
# ========================================================================================================
saturation_factors = [1, 0.5]
noise_levels = [5, 10]
green_factor = 0

noise_types = []
for s in saturation_factors:
    for n in noise_levels:
        noise_types.append("{}_{}_{}/".format(int(s*100), n, green_factor))

base_noise_folder = '/home/ben/image_enhancement/experiment_images/noisy_'
for noise_setting in noise_types:
    noise_folder = base_noise_folder + noise_setting
    evaluate_images(noise_folder, method_folders, plotstuff=False)

print "Evaluated the methods"
