import csv
import os
from collections import defaultdict


def calculate_section_averages(input_file, output_file):
    section_scores = defaultdict(list)  # Przechowuje wyniki dla każdej sekcji

    current_section = None

    with open(input_file, 'r') as infile:
        for line in infile:
            line = line.strip()

            # Sprawdź, czy linia oznacza nową sekcję (np. "# 1")
            if line.startswith('#'):
                current_section = line
                continue

            # Pomijamy linie nagłówka i puste linie
            if line == "time,name,score" or not line:
                continue

            # Parsujemy dane
            try:
                time, name, score = line.split(',')
                score = float(score)
                if current_section is not None:  # Tylko jeśli jesteśmy w sekcji
                    section_scores[current_section].append(score)
            except ValueError:
                continue

    # Oblicz średnie dla każdej sekcji
    section_averages = {}
    for section, scores in section_scores.items():
        if scores:  # Unikaj dzielenia przez zero
            section_averages[section] = sum(scores) / len(scores)

    # Zapisz wyniki do nowego pliku
    with open(output_file, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Section', 'Average Score'])
        for section, avg in sorted(section_averages.items(), key=lambda x: x[0]):
            writer.writerow([section, f"{avg:.2f}"])

    return section_averages


def process_directory(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join("test_results/avg", f'avg_{filename}')

            averages = calculate_section_averages(input_path, output_path)
            results[filename] = averages
            print(f"Processed {filename}:")
            for section, avg in averages.items():
                print(f"  {section}: {avg:.2f}")
            print()

    return results


if __name__ == '__main__':
    directory = "test_results/cleaned"
    process_directory(directory)