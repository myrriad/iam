from pdf2image import convert_from_path
from PIL import Image

# get images from pdf
def extract_from_pdf(pdf_path, image_path):
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        image.save(image_path + str(i) + '.png', 'PNG')

# only do this if path doesn't exist

# extract_from_pdf('looseleaf/pdfs/Blank1.pdf', 'looseleaf/pngs/Blank1/')

# load images
img1 = Image.open('looseleaf/pngs/Blank1/0.png')
img2 = Image.open('iamdataset/words-sample/a01/a01-007/a01-007-00-00.png')

# display images for debug
img1.show()
img2.show()

# overlay img2 onto img1 by make img1 transparent
# make both RGBA
img1 = img1.convert("RGBA")
img2 = img2.convert("RGBA")
img1.paste(img2, (0, 0), img2)
img1.show()


# for the data production stage, we take the iam dataset and overlay the looseleaf dataset on top of it
# the combined is like looseleaf with writing

# to recover the writing, removing the lines,
# for simplicity let's have the model generate a map of possibly transparent,
# possibly fully opaque pixels
# and then overlay the generated image over the combined writing.
# the hope is that the model will generate pixels that exactly cover the lines
# so that the final result is exactly the writing


# alternatively image subtraction
# https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html
# https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# the idea is to have the model generate a mask of the lines, assuming that lines are easier to detect
# then take the inverse of the mask should extract only the writing
#
