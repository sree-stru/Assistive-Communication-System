from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import numpy as np

# INPUT DATASET
input_base = r"C:\Users\LVRSS\Assistive-Communication-System\backend\images for phrases\images for phrases"

# OUTPUT DATASET
output_base = r"C:\Users\LVRSS\Assistive-Communication-System\backend\augmented_phrases"

os.makedirs(output_base, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.8,1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

for phrase in os.listdir(input_base):

    input_folder = os.path.join(input_base, phrase)
    output_folder = os.path.join(output_base, phrase)

    os.makedirs(output_folder, exist_ok=True)

    for img_name in os.listdir(input_folder):

        img_path = os.path.join(input_folder, img_name)

        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        i = 0
        for batch in datagen.flow(
            x,
            batch_size=1,
            save_to_dir=output_folder,
            save_prefix='aug',
            save_format='jpg'
        ):
            i += 1
            if i >= 10:   # create 10 images from each image
                break