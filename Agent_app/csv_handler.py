import pandas as pd
import json

class CSVHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.results = []
        self.full_text = ""     # full CSV text

        #  initilize the read csv
        self.read_csv()
    def read_csv(self):
        """Reads a CSV file and returns a DataFrame."""
        try:
            df = pd.read_csv(self.file_path)

            # Convert to JSONL (one JSON object per line)
            # self.full_text = '\n'.join(
            #     [json.dumps(record, ensure_ascii=False) for record in df.to_dict(orient='records')]
            # )

            # Convert to compact table text (CSV format)
            self.full_text = df.to_csv(index=False, sep="|")

            
            return self.full_text
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

if __name__ == "__main__":
    csv_handler = CSVHandler("../media/uploads/program_abstract_20251022_071249.csv")
    # full_text = csv_handler.read_csv()
    # print(type(full_text))
    # if full_text:
    #     print("Full CSV Text:")
    #     print(full_text)

    # #  chunk the data
    if csv_handler.full_text is not None:
        # Convert DataFrame to string for chunking
        print("Full CSV DataFrame:")
        print(csv_handler.full_text.head())  # Print first few rows
