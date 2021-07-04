import os
import random
from PIL import Image, ImageDraw

INPUT_FILE_PATH = './yolo/training/data.csv'
OUTPUT_FILE_PATH = './yolo/training/data_adapted.csv'
IMAGE_ROOT_DIR = './'


TRAINING_OUTPUT_FILE = './yolo/training/training_data.csv'
TEST_OUTPUT_FILE = './yolo/training/test_data.csv'
bounding_boxes = {}

"""
    This function is used to convert csv exported from google sheet to
    one suitable for yolo training.
"""
def google_sheet_csv_to_yolo_csv(input_file_path=INPUT_FILE_PATH, output_file_path=OUTPUT_FILE_PATH, image_root_dir=IMAGE_ROOT_DIR):

    with open(input_file_path, 'r') as input_file_stream:
        lines = input_file_stream.readlines()

    with open(output_file_path, 'w') as output_file_stream:
        for line in lines:
            values = google_sheet_csv_line_to_yolo_values(line)
            image, rest = values[0] + '.jpg', values[1:]

            bounding_boxes[image] = []

            output_file_stream.write(image_root_dir + image + ' ')

            num_of_bounding_boxes = int(len(rest) / 5)
            for i in range(num_of_bounding_boxes):
                bounding_boxes[image].append(rest[i * 5: i * 5 + 5])
                output_file_stream.write(','.join(rest[i*5: i*5+5]))
                if i < num_of_bounding_boxes - 1:
                    output_file_stream.write(" ")

            output_file_stream.write('\n')


def google_sheet_csv_line_to_yolo_values(line):
    values = line.replace(',', ' ').replace('"', '').strip().split(' ')

    if len(values) % 5 != 1:  # one because of image on 0.
        print('Not enough values on line:', line)

    for i in range(1, len(values)):
        if i % 5 in [3, 4]:  # adding top_x + width / top_y + height
            values[i] = str(int(values[i-2]) + int(values[i]))

        if i % 5 in [1, 2, 3, 4]:
            if int(values[i]) < 0 or int(values[i]) > 800:
                print('Coordinates must be in range [0, 800]. (', line.strip(), ')')

        if i % 5 in [0]:
            if values[i] not in ['0', '1']:
                print('Class must be \'0\' or \'1\'. (', line.strip(), ')')

    return values


"""
    This function is used to draw all bounding boxes of models on images
    for training so that we can manually check dataset.
"""
def images_bounder(images_dir, images_dir_bounded=None):
    if images_dir_bounded is None:
        images_dir_bounded = images_dir + 'bounded/'

    if not os.path.isdir(images_dir_bounded):
        os.mkdir(images_dir_bounded)

    for filename in os.listdir(images_dir):
        if not filename.endswith('.jpg'):
            print(filename, 'is not an image.')
            continue

        image = Image.open(images_dir + filename)
        draw = ImageDraw.Draw(image)

        for bounding_box in bounding_boxes[filename]:
            bounding_box_top = (int(bounding_box[0]), int(bounding_box[1]))
            bounding_box_bottom = (int(bounding_box[2]), int(bounding_box[3]))
            bounding_box_color = 'blue' if bounding_box[4] == '0' else 'red'
            draw.rectangle([bounding_box_top, bounding_box_bottom], outline=bounding_box_color)

        image.save(images_dir_bounded + filename[:-4] + '_bounded.jpg')


def data_separator(input_file_path=INPUT_FILE_PATH, training_output_file=TRAINING_OUTPUT_FILE, test_output_file=TEST_OUTPUT_FILE, test_set=10):
    with open(input_file_path, 'r') as input_file_stream:
        lines = input_file_stream.readlines()

    test_lines = random.sample(lines, int(len(lines)*test_set/100))

    for test_line in test_lines:
        lines.remove(test_line)

    with open('./test_data_temporary.csv', 'w') as test_output_temp_file:
        test_output_temp_file.writelines(test_lines)

    with open('./training_data_temporary.csv', 'w') as training_output_temp_file:
        training_output_temp_file.writelines(lines)

    google_sheet_csv_to_yolo_csv('./test_data_temporary.csv', test_output_file)
    google_sheet_csv_to_yolo_csv('./training_data_temporary.csv', training_output_file)

    os.remove('./test_data_temporary.csv')
    os.remove('./training_data_temporary.csv')


if __name__ == "__main__":
    print("RECS1.6 Projekat by ~Serbedzija, ~Panic")

    #google_sheet_csv_to_yolo_csv()
    data_separator()
    # images_bounder('C:/Games/HLAE/untitled_rec/images/')
