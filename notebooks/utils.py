import glob
import shutil
import matplotlib.pyplot as plt
import random
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import kagglehub

def image_dimensions(image_dataset, chosen_height = 50, chosen_width = 50):
    """ Creating a function that plots heights and widths of a picture dataset
    And also returns the values of unique heights and widths

    """
    heights = []
    widths = []

    # Looping on the images to gather height and width data
    for image in image_dataset:

        # Load the image and convert it to an array
        sample_image = load_img(image)
        sample_image = img_to_array(sample_image)
        
        # Get the width and height and add to collections
        heights.append(sample_image.shape[0])
        widths.append(sample_image.shape[1])

    # Plotting results on a histogram
    plt.hist(heights, bins = 50, alpha = 0.5, label = "Image Heights")
    plt.hist(widths,  alpha = 0.5, label = "Image Widths")
    plt.xlabel('Image Dimension')
    plt.ylabel('Count')
    plt.legend(loc = 'best')
    plt.show()

    # Check if all heights are the chosen height
    unique_heights = set(heights)

    if unique_heights == {chosen_height}:
        print(f"All heights are {chosen_height}.")
    else:
        print(f"There are heights that are not equal to {chosen_height}.")
        print("Unique heights:", unique_heights)

    
    # Check if all widths are the chosen height
    unique_widths = set(widths)

    if unique_widths == {chosen_height}:
        print(f"All widths are {chosen_height}.")
    else:
        print(f"There are widths that are not equal to {chosen_height}.")
        print("Unique widths:", unique_widths)


def plot_image_samples(positive_images, negative_images, rows=3, cols=2, figsize=(6, 4)):
    """ Defining a function that plots images to inspect:

    """

    # Create the subplot grid
    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    # Zip positive and negative images with the subplot axes
    for positive, negative, row in zip(positive_images[:rows], negative_images[:rows], axs):
        for pic, ax in zip([positive, negative], row):
            img = load_img(pic)  # Load the image
            img_array = img_to_array(img)  # Convert to an array
            img_array /= 255.0  # Normalize the image
            ax.imshow(img_array)  # Display the image
            ax.axis('off')  # Remove axis for better visualization

    plt.tight_layout()
    plt.show()