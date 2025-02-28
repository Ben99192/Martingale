from blackwater.utility.streaming.iter_csv import iter_csv
import os
import csv


class HeightWeightDataset:
    """
    A simple dataset class to load and iterate over the height-weight dataset.

    - Automatically loads data from the 'Martingale/Dataset/' directory.
    - Reads a CSV file without requiring FileDataset.
    - Converts gender (Male/Female) into numerical values (Male=1, Female=0).
    - Provides an iterator to yield data row by row.
    """

    def __init__(self, filename="weight-height.csv"):
        """
        Initialize the dataset by loading the CSV file.

        :param filename: Name of the CSV file (not full path).
        """
        base_dir = os.path.join(os.path.dirname(__file__), "..", "Dataset")
        self.file_path = os.path.abspath(os.path.join(base_dir, filename))

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        self.data = self._load_data()

    def _load_data(self):
        """Load the CSV file into a list of dictionaries."""
        data = []
        with open(self.file_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                row["Gender"] = 1 if row["Gender"].strip().lower() == "male" else 0
                row["Height"] = float(row["Height"])
                row["Weight"] = float(row["Weight"])
                data.append(row)
        return data

    def __iter__(self):
        """Make the dataset iterable."""
        return iter(self.data)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
