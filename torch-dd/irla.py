import pandas as pd

url = 'https://gairuo.com/file/data/dataset/iris.data'
df = pd.read_csv(url)
df.head()