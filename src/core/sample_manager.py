from data_io.excel_reader import load_excel
from core.sample import Sample


class SampleManager:
    def __init__(self):
        self.data, self.samples = None, []

    def load_excel(self, filename):
        self.data = load_excel(filename)
        self.samples = [Sample.from_dict(row.to_dict()) for _, row in self.data.iterrows()]
        return self.samples

    def get_sample(self, name):
        return next((sample for sample in self.samples if sample.name == name), None)