import tushare as ts
import pandas as pd

####常量设置####
fileName = '/Users/liuweidong/Desktop/LSTM_STOCK/data/'

####链接地址的token,'a82de...............................'这部分是密钥####
pro = ts.pro_api('f23caad04846ebbd2ec812d0f043cdee5dbd4e257e95b17a81e318ee')

###下载个股信息,以000001.SZ,600000.SH为例
df = pro.daily(ts_code='000010.SZ', start_date='20120501', end_date='20220501')
df.index.name='id'

###按时间升序，index降序
df=df.sort_index(ascending=False)
df.to_csv(fileName+'000001SZ_daily.csv')

