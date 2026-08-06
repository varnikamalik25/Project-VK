
import pyspng

def decode_image(image_path):
    '''Decodes PNG images to numpy arrays'''
    with open(image_path, "rb") as f:
        img_array = pyspng.load(f.read())
    return img_array



