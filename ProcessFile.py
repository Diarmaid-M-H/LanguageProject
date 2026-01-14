import csv
# Script to process the 2022 census data
# Problem: 2022 data is not grouped by county as 2006,2011,2016 data is.
# Solution: Script to go through file and select all the rows which are present in 2016 file
# Key: compare the "GEOGDESC" column. Cannot use GUID as GUID was changed between the censuses.

def load_csv(file_path):
    # Load CSV file and return list of dictionaries.
    with open(file_path, 'r', newline='', encoding='iso-8859-1') as file:
        reader = csv.DictReader(file)
        return list(reader)


def write_csv(data, file_path):
    # Write list of dictionaries to a CSV file.
    with open(file_path, 'w', newline='', encoding='iso-8859-1') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def process_files(file_2022, file_2016, output_file):
    """Process SAPS_2022_RAW.csv based on SAPS_2016.csv and write output to output_file."""
    data_2022 = {row['GUID'].replace('-', '').lower(): row for row in load_csv(file_2022)}
    with open(file_2016, 'r', newline='', encoding='iso-8859-1') as file_2016:
        reader_2016 = csv.DictReader(file_2016)
        header = reader_2016.fieldnames
        processed_data = []
        for idx, row_2016 in enumerate(reader_2016):
            guid_2016 = row_2016['GUID'].replace('-', '').lower()
            if guid_2016 in data_2022:
                processed_data.append(data_2022[guid_2016])
            else:
                print("ROW NOT FOUND for GUID:", row_2016['GUID'])
    write_csv(processed_data, output_file)

if __name__ == "__main__":
    SAPS_2022_file = "Output CSVs/SAPS_2022_RAW.csv"
    SAPS_2016_file = "Output CSVs/SAPS_2016.csv"
    output_file = "Output CSVs/SAPS_2022.csv"
    process_files(SAPS_2022_file, SAPS_2016_file, output_file)
