def google_sheet_csv_to_yolo_csv(input_file_path='./training/data.csv', output_file_path='./training/data_adapted.csv', image_root_dir='./'):

    with open(input_file_path, 'r') as input_file_stream:
        lines = input_file_stream.readlines()

    with open(output_file_path, 'w') as output_file_stream:

        for line in lines:
            # TODO: validate each line

            line = line.replace(',', ' ').replace('"', '').strip()
            image = line.split(' ')[0]
            output_file_stream.write(image_root_dir + image + '.jpg ')

            rest = line.split(' ')[1:]
            for i in range(int(len(rest) / 5)):
                output_values = [
                        rest[i*5],
                        rest[i*5 + 1],
                        str(int(rest[i*5]) + int(rest[i*5+2])),
                        str(int(rest[i*5+1]) + int(rest[i*5+3])), rest[i*5+4],
                        ''
                    ]
                output_file_stream.write(' '.join(output_values))

            output_file_stream.write('\n')


if __name__ == "__main__":
    print("RECS1.6 Projekat")

    google_sheet_csv_to_yolo_csv()

