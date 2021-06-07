from io import BytesIO
from flask import send_file
import base64
from PIL import Image
from argparse import ArgumentParser

def get_arguments():
    parser = ArgumentParser()

    parser.add_argument("img_path_1", type=str, help="File path to first image")
    parser.add_argument("img_path_2", type=str, help="File path to second image")
    parser.add_argument("output_path", type=str, help="Output file path")
    parser.add_argument("--gif", action="store_true", help="Whether generate morhping sequence")
    parser.add_argument("--image_size", type=int, help="Size of cropped and resized face", default=256)
    parser.add_argument("--predictor_path", type=str, 
        help="File path to pretrained face feature detection model", default="pretrained/shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--alpha", type=float, help="Alpha for merged image", default=0.5)
    parser.add_argument("--bokeh", action="store_true", help="Whether to use bokeh effect")
    parser.add_argument("--delaunay", action="store_true", help="Whether to use Delaunay triangulation for feature line extraction")
    parser.add_argument("--a", type=float, help="Parameter (a) in paper", default=1)
    parser.add_argument("--b", type=float, help="Parameter (b) in paper", default=2)
    parser.add_argument("--p", type=float, help="Parameter (p) in paper", default=0.5)
    parser.add_argument("--steps", type=int, help="Morphing steps in morph sequence (for gif generation)", default=10)
    parser.add_argument("--duration", type=int, help="Morph sequence gif duration (for gif generation)", default=5)

    args = parser.parse_args()
    return args

def b64ToImage(b64_string):
    data = base64.b64decode(b64_string)
    buf = BytesIO(data)
    img = Image.open(buf)
    return img 
    
def serve_pil_image(pil_img, ext):
    """ serve a PIL image file as a flask response
    
    Args: 
        pil_img: PIL img file to serve.
        ext: file extension
    Returns:
        A flask send_file response.
    """
    img_io = BytesIO()
    ext = ext.lower()  # normalize extension
    pil_img.save(img_io, ext, quality = 75) #  default quality
    img_io.seek(0) #  go to head of io file
    return base64.b64encode(img_io.getvalue())

def get_name_from_file(filename):
    name = filename.split(".")[0]
    name = name.split("_")
    first_name = name[0].capitalize()
    last_name = name[1].capitalize()
    return " ".join([first_name, last_name])
