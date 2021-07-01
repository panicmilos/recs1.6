INPUT_FILE_PATH = './yolo/training/data.csv'
OUTPUT_FILE_PATH = './yolo/training/data_adapted.csv'
IMAGE_ROOT_DIR = './'


def google_sheet_csv_to_yolo_csv(input_file_path=INPUT_FILE_PATH, output_file_path=OUTPUT_FILE_PATH, image_root_dir=IMAGE_ROOT_DIR):

    with open(input_file_path, 'r') as input_file_stream:
        lines = input_file_stream.readlines()

    with open(output_file_path, 'w') as output_file_stream:

        for line in lines:
            # TODO: validate each line

            line = line.replace(',', ' ').replace('"', '').strip()
            image = line.split(' ')[0]
            output_file_stream.write(image_root_dir + image + '.jpg ')

            rest = line.split(' ')[1:]
            if len(rest) % 5 != 0:
                print(image)

            num_of_bounding_boxes = int(len(rest) / 5)
            for i in range(num_of_bounding_boxes):
                output_values = [
                        rest[i*5],
                        rest[i*5 + 1],
                        str(int(rest[i*5]) + int(rest[i*5+2])),
                        str(int(rest[i*5+1]) + int(rest[i*5+3])), rest[i*5+4]
                    ]

                output_file_stream.write(','.join(output_values))
                if i < num_of_bounding_boxes - 1:
                    output_file_stream.write(" ")

            output_file_stream.write('\n')


if __name__ == "__main__":
    print("RECS1.6 Projekat")

    google_sheet_csv_to_yolo_csv()

