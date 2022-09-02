import json
import pandas as pd
from pandas import DataFrame

class Config():
    def __init__(self):
        with open("config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)
            f.close()

#TODO: when filepath source change to config file, change static method to class method
class DataIO():
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def readCSV(filename) -> DataFrame:
        data = pd.read_csv(filename)
        data.index = pd.DatetimeIndex(data['Date'])
        return data

    @staticmethod
    def writeCSV(self, filename, data):
        data.to_csv(filename)