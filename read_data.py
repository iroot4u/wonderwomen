import pandas as pd
f = pd.read_csv('data/train.csv')
#/Applications/PyCharm CE.app/Contents/helpers/pydev/pydevconsole.py:81: DtypeWarning: Columns (12,49,51,56,91,126,145,163,165,166,167,168,169,171,173,174,176,177,242,244,246,248,252,255,274,290,291,292,294,295,296,332,344,366,374,376,397,414,440,491,620,634,639,642,643,645,710,713,760,769,810,829,929,954,979,1001,1002,1003,1004,1005,1024,1037,1041,1043,1062,1086,1099,1100,1121,1129,1136,1152,1153,1166,1168,1182,1193,1204,1205,1207,1208,1216,1226,1228,1230,1232,1234) have mixed types. Specify dtype option on import or set low_memory=False.
#self.more = self.interpreter.runsource(text, '<input>', symbol)

f = pd.read_csv('data/train.csv')
f.head()
f.describe()
f2 = f.dropna()
f2.describe()
len(f)
len(f2)
f3 = dropna(axis=1)
f3=f.dropna(axis=1)
len(f3)
len(f3.columns)
len(f.columns)

