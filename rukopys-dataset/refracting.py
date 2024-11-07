import os
from PIL import Image, ImageOps, ImageEnhance
import pandas as pd
import numpy as np

input_folder = './rukopys-dataset/glyphs_200x200'
resized_folder = './rukopys-dataset/glyphs_28x28'
output_csv = './data/ukr_mnist_data.csv'
glyphs_df = pd.read_csv("./rukopys-dataset/glyphs.csv")
glyphs_map = {}
for index, row in glyphs_df.iterrows():
    glyphs_map[row['filename']] = {
        'label': row['label'],
        'is_uppercase': row['is_uppercase']
    }

def resize_images(input_folder, output_folder, size=(28, 28), contrast_factor=10.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)  # Convert to grayscale

            img_contrasted = ImageEnhance.Contrast(img).enhance(contrast_factor)

            #img_resized = img.resize(size, resample=Image.Resampling.BILINEAR)
            img_resized = img_contrasted.resize(size, resample=Image.Resampling.BICUBIC)
            
            resized_img_path = os.path.join(output_folder, filename)
            img_resized.save(resized_img_path)

def images_to_dataframe(image_folder):
    x_data = []
    y_data = []

    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert('L')


            img_inverted = ImageOps.invert(img)
            pixels = np.array(img_inverted).flatten()

            # Extract the first letter of the filename for the label (y_data)
            glyph_info = glyphs_map.get(f"glyphs/{filename}")
            if glyph_info:
                label = glyph_info['label']
                if glyph_info['is_uppercase']:
                    continue
                    label = label.upper()
                y_data.append(label)
                x_data.append(pixels)
            else:
                continue
                y_data.append(filename[0])

    x_data = np.array(x_data).reshape(-1, 28, 28)
    y_data = np.array(y_data)

    return x_data, y_data


resize_images(input_folder, resized_folder)
x_data, y_data = images_to_dataframe(resized_folder)

df = pd.DataFrame({
    'label': y_data,
    'image_data': [x.flatten().tolist() for x in x_data]
})
df.to_csv(output_csv, index=False)