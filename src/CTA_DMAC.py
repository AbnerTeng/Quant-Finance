from backtesting import Backtest, Strategy #引入回測和交易策略功能

from backtesting.lib import crossover #從lib子模組引入判斷均線交會功能
from backtesting.test import SMA #從test子模組引入繪製均線功能
from backtesting.test import GOOG #從test子模組引入GOOG
import pandas as pd #引入pandas讀取股價歷史資料CSV檔
import sys
from make_df import load_data

class SmaCross(Strategy):
    def init(self):
        self.fast_line = self.I(SMA, self.data.Close, 13)
        self.slow_line = self.I(SMA, self.data.Close, 48)

    def next(self):
        if crossover(self.fast_line, self.slow_line):
            print(
                f"{self.data.index[-1]} Buy: Price: {self.data.Close[-1]}, Slow: {self.slow_line[-5:]}, Fast: {self.fast_line[-5:]}"
            )
            self.buy()
        elif crossover(self.slow_line, self.fast_line):
            print(
                f"{self.data.index[-1]} Sell: Price: {self.data.Close[-1]}, Slow: {self.slow_line[-5:]}, Fast: {self.fast_line[-5:]}"
            )

            self.sell()

def stock_no():
    stock = load_data()
    return stock

## stock_no = GOOG   
##if(sys.argv[1] != None):
##    stock_no = sys.argv[1]

#df = pd.read_csv(f"{stock_no}.csv", index_col=0) #pandas讀取資料，並將第1欄作為索引欄
# df = df.interpolate() #CSV檔案中若有缺漏，會使用內插法自動補值，不一定需要的功能
#df.index = pd.to_datetime(df.index) #將索引欄資料轉換成pandas的時間格式，backtesting才有辦法排序

test = Backtest(
    stock_no,
    SmaCross,
    cash=1000000,
    commission=0.004,
    exclusive_orders=True,
    trade_on_close=True,
)
# 指定回測程式為test，在Backtest函數中依序放入(資料來源、策略、現金、手續費)

result = test.run()
#執行回測程式並存到result中

print(result) # 直接print文字結果
test.plot(filename=f"res{stock_no}.html") #將線圖網頁依照指定檔名保存