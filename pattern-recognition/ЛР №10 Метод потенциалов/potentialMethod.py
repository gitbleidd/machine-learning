import os
import pickle

from PIL import Image
from multiprocessing import Pool

train_dir = os.path.join(os.getcwd(), "prepared_train")
test_dir = os.path.join(os.getcwd(), "prepared_test")


def main():
    train_maps = {}  # Contains every train pattern path, potential map.
    processes_data = []  # Contains image_path, pattern_name

    maps_file_path = os.path.join(os.getcwd(), "train_maps.pickle")
    if os.path.isfile(maps_file_path):
        with open(maps_file_path, 'rb') as f:
            train_maps = pickle.load(f)
        print("Done loading dump")
    else:
        for pattern_name in os.listdir(train_dir):
            pattern_folder_path = os.path.join(train_dir, pattern_name)
            train_maps[pattern_name] = []
            for image_name in os.listdir(pattern_folder_path):
                image_path = os.path.join(pattern_folder_path, image_name)
                processes_data.append((image_path, pattern_name))

        with Pool(processes=os.cpu_count()) as pool:
            res = pool.map(process_train_images_map, processes_data)
            for item in res:
                image_path = item[0]
                pattern_name = item[1]
                pot_map = item[2]
                train_maps[pattern_name].append((image_path, pot_map))
            print("Done processing train images maps")

        # Making potential maps dump
        with open(maps_file_path, 'wb') as f:
            pickle.dump(train_maps, f)

    processes_data = []  # Contains image_path, pattern_name, train_maps
    compare_result = {}
    for pattern_name in os.listdir(test_dir):
        pattern_folder_path = os.path.join(test_dir, pattern_name)
        compare_result[pattern_name] = []
        for image_name in os.listdir(pattern_folder_path):
            image_path = os.path.join(pattern_folder_path, image_name)
            processes_data.append((image_path, pattern_name, train_maps))
            #break # DEBUG break (not full calculations)

    with Pool(processes=os.cpu_count()) as pool:
        res = pool.map(process_test_images, processes_data)
        for item in res:
            train_image_path = item[0]
            pattern_name = item[1]

            min_dist_key = item[2]
            min_dist_pattern = item[3]

            res_pattern_name = min_dist_key
            res_img_path = min_dist_pattern[0]
            compare_result[pattern_name].append((train_image_path, res_pattern_name, res_img_path))
        print("Done comparing train/test images")

    open("results.txt", "w").close()  # Clear file
    f = open("results.txt", "a")

    conf_matrix = {}
    pattern_labels = [str(chr(i)) for i in range(ord('A'), ord('Z') + 1)] + [str(chr(i)) for i in range(ord('0'), ord('9') + 1)]

    f.write("In pattern | Recognized pattern | In path | Recognized path\n")
    for key, value in compare_result.items():
        conf_matrix[key] = {}
        for label in pattern_labels:
            conf_matrix[key][label] = 0

        for p in value:
            f.write("{0} {1} {2} {3}\n".format(key, p[1], p[0], p[2]))
            conf_matrix[key][p[1]] += 1

    f.write("\n\n Confusion matrix\n  ")
    for lableName in pattern_labels:
        f.write(lableName)
        f.write(" ")
    f.write("\n")

    for lineName in pattern_labels:
        line = conf_matrix[lineName]
        f.write(lineName)
        f.write(" ")
        for col_name in pattern_labels:
            cell_value = line[col_name]
            f.write(str(cell_value))
            f.write(" ")
        f.write("\n")

    f.close()

    # Dump method results
    conf_matrix_path = os.path.join(os.getcwd(), "conf_matrix.pickle")
    with open(conf_matrix_path, 'wb') as f:
        pickle.dump(conf_matrix, f)

    potential_res_path = os.path.join(os.getcwd(), "potential_res.pickle")
    with open(potential_res_path, 'wb') as f:
        pickle.dump(compare_result, f)


def process_train_images_map(image):
    image_path = image[0]
    pattern_name = image[1]
    pot_map = calc_potential_map(image_path)

    return (image_path, pattern_name, pot_map)


def calc_potential_map(img_path):
    img = Image.open(img_path)

    width = img.size[0]
    height = img.size[1]
    pot_map = [[0] * width for p in range(height)]

    coef2 = 1/6
    coef3 = 1/12
    for i in range(height):
        for j in range(width):
            # +1, если точка (x,y) на исходном изображении черная
            # if img.getpixel((i, j)) == 255:
            #     pot_map[i][j] = 0
            if img.getpixel((i, j)) == 0:
                pot_map[i][j] = 1

                # +1/6 на каждую точку из точек исходного изображения
                # (x-1,y),(x,y+1),(x+1,y),(x,y1), если эта точка черная
                if i != 0:
                    pot_map[i - 1][j] += coef2
                if i != (height - 1):
                    pot_map[i + 1][j] += coef2
                if j != 0:
                    pot_map[i][j - 1] += coef2
                if j != (width - 1):
                    pot_map[i][j + 1] += coef2

                # +1/12 на каждую точку из точек исходного изображения
                # (x-1,y-1),(x-1,y+1),(x+1,y+1),(x+1,y-1), если эта точка черная
                if i != 0 and j != 0:
                    pot_map[i - 1][j - 1] += coef3
                if i != 0 and j != (width - 1):
                    pot_map[i - 1][j + 1] += coef3
                if i != (height - 1) and j != (width - 1):
                    pot_map[i + 1][j + 1] += coef3
                if i != (height - 1) and j != 0:
                    pot_map[i + 1][j - 1] += coef3
    return pot_map


def calc_test_train_distance(test_map, train_map):
    height = len(test_map)
    width = len(test_map[0])

    dist_sum = 0
    for i in range(height):
        for j in range(width):
            dist_sum += (test_map[i][j] - train_map[i][j]) ** 2

    return dist_sum ** 0.5


def process_test_images(image):
    image_path = image[0]
    pattern_name = image[1]
    train_maps = image[2]

    test_map = calc_potential_map(image_path)  # potential map for test image

    # Сравниваем со всеми тренировачными картами
    min_dist = float("inf")
    min_dist_pattern = None
    min_dist_key = None
    for key, value in train_maps.items():
        for p in value:
            dist = calc_test_train_distance(test_map, p[1])
            if dist < min_dist:
                min_dist = dist
                min_dist_pattern = p
                min_dist_key = key

    return (image_path, pattern_name, min_dist_key, min_dist_pattern)