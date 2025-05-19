import csv
import os


def clean_csv(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, \
            open(output_file, 'w', newline='') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)

        writer.writeheader()

        for row in reader:
            if not (row['name'] == 'Unknown' and row['score'] == '0.0'):
                writer.writerow(row)


def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, "cleaned",f'cleaned_{filename}')

            clean_csv(input_path, output_path)
            print(f'Processed: {filename}')


if __name__ == '__main__':
    directory = "test_results"
    process_directory(directory)