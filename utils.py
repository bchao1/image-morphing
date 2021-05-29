from io import BytesIO
from flask import send_file
import base64
from PIL import Image

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
