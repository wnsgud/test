from PIL import Image
import numpy as np

img = Image.open("./temp_data/kaejangcon1.png")

img = img.resize((100,100),Image.ANTIALIAS)

print(img.size)

