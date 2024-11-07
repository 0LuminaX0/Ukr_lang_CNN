# Removing sample data and loading actual input CSV

import ast
import pandas as pd
import numpy as np
from PIL import Image

# Rotation and scaling combinations
rotations = [-20, -10, 0, 10, 20]
scales = [(1, 1), (1, 0.8), (0.8, 1), (1, 1.2), (1.2, 1)]
canvas_size = (28, 28)

input_csv = './data/ukr_mnist_data.csv'
output_csv = './data/augmented_glyphs_2.csv'

def load_data_from_csv(csv_file):
    """Load CSV data, converting the 'image_data' string lists back into arrays."""
    df = pd.read_csv(csv_file)
    y_data = df['label'].values
    x_data = df['image_data'].apply(lambda x: ast.literal_eval(x)).values
    x_data = np.array(x_data.tolist()).reshape(-1, 28, 28)
    return x_data, y_data

def augment_image(image_array, label):
    """Generate augmented images by rotating and scaling each image."""
    augmented_images = []
    image = Image.fromarray(np.array(image_array, dtype=np.uint8).reshape(canvas_size)).convert("L")
    
    for angle in rotations:
        rotated_img = image.rotate(angle, resample=Image.BICUBIC)
        for scale_x, scale_y in scales:
            new_size = (int(canvas_size[0] * scale_x), int(canvas_size[1] * scale_y))
            resized_img = rotated_img.resize(new_size, resample=Image.BICUBIC)

            centered_img = Image.new("L", canvas_size, color=0)
            paste_x = max((canvas_size[0] - new_size[0]) // 2, 0)
            paste_y = max((canvas_size[1] - new_size[1]) // 2, 0)
            centered_img.paste(resized_img, (paste_x, paste_y))
            
            final_img_data = np.array(centered_img).flatten().tolist()
            augmented_images.append((label, final_img_data))
    
    return augmented_images


x_data, y_data = load_data_from_csv(input_csv)

augmented_data = []
for image_array, label in zip(x_data, y_data):
    augmented_images = augment_image(image_array, label)
    augmented_data.extend(augmented_images)

augmented_df = pd.DataFrame(augmented_data, columns=['label', 'image_data'])
augmented_df.to_csv(output_csv, index=False)

augmented_df.head()
