INPUT_FILE_PATH = './yolo/training/data.csv'
OUTPUT_FILE_PATH = './yolo/training/data_adapted.csv'
IMAGE_ROOT_DIR = './'


def google_sheet_csv_to_yolo_csv(input_file_path=INPUT_FILE_PATH, output_file_path=OUTPUT_FILE_PATH, image_root_dir=IMAGE_ROOT_DIR):

    with open(input_file_path, 'r') as input_file_stream:
        lines = input_file_stream.readlines()

    with open(output_file_path, 'w') as output_file_stream:
        for line in lines:
            values = google_sheet_csv_line_to_yolo_values(line)
            image, rest = values[0], values[1:]

            output_file_stream.write(image_root_dir + image + '.jpg ')

            num_of_bounding_boxes = int(len(rest) / 5)
            for i in range(num_of_bounding_boxes):
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


if __name__ == "__main__":
    print("RECS1.6 Projekat by ~Serbedzija, ~Panic")

    google_sheet_csv_to_yolo_csv()

