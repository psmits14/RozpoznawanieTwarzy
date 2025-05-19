import os
import csv
import re
from collections import defaultdict


def extract_name(filename):
    """Wyodrębnia NAME z nazwy pliku w formacie avg_cleaned_NAME_log.csv"""
    match = re.search(r'avg_cleaned_(.*?)_log\.csv', filename)
    return match.group(1) if match else filename


def process_file(input_file, results_dict):
    section_scores = {}

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()

            if line.startswith('#'):
                try:
                    section_raw, score = line[1:].strip().split(',')
                    section_number = int(section_raw.strip())
                    score = float(score)
                    section_scores[section_number] = score
                except ValueError:
                    continue

    file_name = extract_name(os.path.basename(input_file))

    for section, score in section_scores.items():
        results_dict[file_name][section] = score


def process_directory(directory, output_file):
    results = defaultdict(dict)

    for filename in os.listdir(directory):
        if filename.endswith('.csv') and filename.startswith('avg_cleaned_'):
            input_path = os.path.join(directory, filename)
            process_file(input_path, results)

    # Zbierz wszystkie unikalne numery sekcji
    all_sections = sorted(set().union(*(d.keys() for d in results.values())))

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')

        # Nagłówki
        writer.writerow(['Filename'] + [f'{sec}' for sec in all_sections])

        # Dane
        for name, sections in sorted(results.items()):
            row = [name]
            for sec in all_sections:
                score = sections.get(sec, '')
                row.append(f"{score:.2f}" if score != '' else '')
            writer.writerow(row)

    print(f"Zapisano do: {output_file}")
    print(f"Przetworzono plików: {len(results)}")


if __name__ == '__main__':
    directory = "test_results/avg"
    output_file = os.path.join(directory, 'combined_results.csv')
    process_directory(directory, output_file)
