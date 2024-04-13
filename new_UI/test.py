import yfinance as yf

df = yf.download(tickers="SBIN.NS",group_by = 'ticker',threads=True,period='max',interval='1d')
df.reset_index(level=0, inplace=True)
print(df)