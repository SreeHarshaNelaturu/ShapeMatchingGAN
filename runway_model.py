from options import TestOptions
import torch
from models import GlyphGenerator, TextureGenerator
from utils import load_image, to_data, to_var, visualize, save_image, gaussian
import os
import runway
from runway.data_types import *
import cv2

@runway.setup(options={"structure_model": file(extension=".ckpt"), "texture_model": file(extension=".ckpt")})
def setup(opts):
    netGlyph = GlyphGenerator(n_layers=6, ngf=32)
    netTexture = TextureGenerator(n_layers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print("Using:",device)
    
    netGlyph.load_state_dict(torch.load(opts["structure_model"], map_location=device))
    netTexture.load_state_dict(torch.load(opts["texture_model"], map_location=device))

    
    netGlyph.to(device).eval()
    netTexture.to(device).eval()

    return {'netGlyph' : netGlyph,
            'netTexture' : netTexture}

command_inputs = {"input_image" : image, "scale" : number(min=0.0, max=1.0, step=0.1)}
command_outputs = {"output_image" : image}


@runway.command("stylize_text", inputs=command_inputs, outputs=command_outputs, description="Stylize Text based on Texture")
def stylize_text(model, inputs):
    
    text_name = inputs["input_image"]
    text = load_image(text_name, 1)
    text = to_var(text)

    label = inputs["scale"]

    text[:,0:1] = gaussian(text[:,0:1], stddev=0.2)

    img_str = model['netGlyph'](text, label*2.0-1.0) 
    img_str[:,0:1] = gaussian(img_str[:,0:1], stddev=0.2)
    result = to_data(model['netTexture'](img_str))
    
    output = save_image(result[0])

    return {"output_image" : output}


if __name__ == '__main__':
    runway.run()
