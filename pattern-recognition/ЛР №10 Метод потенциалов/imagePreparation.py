from PIL import Image, ImageChops, ImageOps, UnidentifiedImageError
import os
import shutil
import math
import numpy as np

in_folder_name = "train"
out_folder_name = "prepared_train"

img_size = 128
included_extensions = ['jpg', 'jpeg', 'bmp', 'png']
result_dir = os.path.join(os.getcwd(), out_folder_name)


def main():
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    else:
        os.makedirs(result_dir, exist_ok=True)

    generate_images()

    image_paths = []
    for fold in os.listdir(result_dir):
        p = os.path.join(result_dir, fold)
        for image in os.listdir(p):
            image_paths.append(os.path.join(p, image))

    n = math.floor(len(image_paths) ** 0.5)
    image = concat_images(image_paths, (img_size, img_size), (n, n))
    image.show()


def generate_images():
    search_dir = os.path.join(os.getcwd(), in_folder_name)

    char_nums = [i for i in range(ord('A'), ord('Z') + 1)] + [i for i in range(ord('0'), ord('9') + 1)]

    for code in char_nums:
        letter = str(chr(code))

        for i, guid in enumerate(os.listdir(search_dir)):
            guid_abs_path = os.path.join(search_dir, guid)

            subject_abs_path = os.path.join(guid_abs_path, letter)
            # print(subject_abs_path)

            if os.path.isdir(subject_abs_path):
                for j, image_name in enumerate(os.listdir(subject_abs_path)):
                    image_path = os.path.join(subject_abs_path, image_name)

                    # Проверка соответствия форматам изображения
                    if any(image_path.endswith(ext) for ext in included_extensions):
                        img = prepare_image(image_path, img_size)

                        if img is None:
                            continue

                        print(image_path)

                        img_save_folder = os.path.join(result_dir, letter)
                        os.makedirs(img_save_folder, exist_ok=True)
                        # print(os.path.join(img_save_folder, str(i + j)))

                        res_path = os.path.join(img_save_folder, f'{str(i + j)}.png')
                        # img.save(res_path, "JPEG", quality=100, optimize=True, progressive=True)
                        img.save(res_path, "PNG")
                        # print()


def prepare_image(img_path, basewidth):
    try:
        img = Image.open(img_path)
    except UnidentifiedImageError:
        return None

    rgb_img = img.convert('RGB')

    #img = img.convert('L')  # или попробовать img.convert('1')
    #img = img.convert('1')
    #img = ImageOps.autocontrast(img)

    img = img.convert("L").point(lambda x: 0 if x < 250 else 255, '1')

    if np.mean(np.array(img)) < 0.5:
        img = ImageChops.invert(img)

    if np.mean(np.array(img)) < 0.05 or np.mean(np.array(img)) > 0.98:
        return None

    a1 = count_white_lines(rgb_img)
    rgb_img = rgb_img.rotate(90)
    a2 = count_white_lines(rgb_img)
    rgb_img = rgb_img.rotate(90)
    a3 = count_white_lines(rgb_img)
    rgb_img = rgb_img.rotate(90)
    a4 = count_white_lines(rgb_img)
    width, height = img.size
    img = img.crop((a4, a1, width - a2, height - a3))

    # basewidth размер квадрата
    #TODO размещение по центру
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
    #img.show()
    return img


def count_white_lines(img):
    width, height = img.size

    a1 = 0
    for h in range(height):
        is_white = True
        for w in range(width):
            pixel = img.getpixel((w, h))
            if not (int(pixel[0]) >= 254 and int(pixel[1]) >= 254 and int(pixel[2]) >= 254):
                is_white = False
                break
        if is_white:
            a1 += 1
        else:
            break
    return a1


# Собирает все изображение в одно
def concat_images(image_paths, size, shape=None):
    # Open images and resize them
    width, height = size
    images = map(Image.open, image_paths)
    images = [ImageOps.fit(image, size, Image.ANTIALIAS)
              for image in images]

    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('RGB', image_size)

    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            image.paste(images[idx], offset)

    return image
