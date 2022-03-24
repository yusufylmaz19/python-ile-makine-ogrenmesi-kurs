#kutuphaneler
from matplotlib.axis import YAxis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import impute

#veri önişleme
#veri yükleme
veriler=pd.read_csv('eksikveriler.csv')

boy=veriler[['boy']]
both=veriler[['boy','kilo']]

#eksik verilerin işlenmesi
#sci-kit learn    
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan , strategy="mean")
yas=veriler.iloc[:,1:4].values
imputer=imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
#encoder kategori>numerics
ulke=veriler.iloc[:,0:1].values

from sklearn import preprocessing

le=preprocessing.LabelEncoder()
# le=le.fit(ulke[:,0])
# ulke[:,0]=le.transform(ulke[:,0]) or 

ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe=preprocessing.OneHotEncoder()

ulke=ohe.fit_transform(ulke).toarray()

print(ulke)
#numpy dizileri dataframe dönüştürme
sonuc=pd.DataFrame(data=ulke , index=range(22),columns=['fr','tr','us'])
print(sonuc)
sonuc2=pd.DataFrame(data=yas , index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)
cinsiyet=veriler.iloc[:,-1].values
sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)


#dataframeleri birleştirme
concat=pd.concat([sonuc,sonuc2,sonuc3],axis=1)
print(concat)


# verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import  train_test_split

x_train,x_test,y_train,y_test=train_test_split(concat,sonuc3,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

print(X_train)