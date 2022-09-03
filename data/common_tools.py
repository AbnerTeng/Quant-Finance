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
        df = pd.read_csv(filename)
        df.index = pd.DatetimeIndex(df['Date'])
        return df

    @staticmethod
    def writeCSV(self, filename, df):
        df.to_csv(filename)