import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc
import pickle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
ar= pd.read_csv('Airline_Reviews.csv',index_col=0)
ar.head()
ar1=ar.drop(['Review_Title','Review','Review Date','Inflight Entertainment','Wifi & Connectivity','Aircraft','Value For Money'],axis=1)
ar1['Overall_Rating']=ar1['Overall_Rating'].replace('n',10)
ar1['Type Of Traveller']=ar1['Type Of Traveller'].fillna(ar1['Type Of Traveller'].mode()[0])
ar1['Seat Type']=ar1['Seat Type'].fillna(ar1['Seat Type'].mode()[0])
ar1['Route']=ar1['Route'].fillna(ar1['Route'].mode()[0])
ar1['Date Flown']=ar1['Date Flown'].fillna(ar1['Date Flown'].mode()[0])
ar1['Seat Comfort']=ar1['Seat Comfort'].fillna(ar1['Seat Comfort'].mode()[0])
ar1['Cabin Staff Service']=ar1['Cabin Staff Service'].fillna(ar1['Cabin Staff Service'].mode()[0])
ar1['Food & Beverages']=ar1['Food & Beverages'].fillna(ar1['Food & Beverages'].mode()[0])
ar1['Ground Service']=ar1['Ground Service'].fillna(ar1['Ground Service'].mode()[0])
ar1['Date Flown']=ar1['Date Flown'].astype(str)
ar1[['Month Flown','Year Flown']]=ar1['Date Flown'].str.extract(r'(\w+)\s(\d+)')
ar1['Origin']=ar1.Route.str.split('to',expand=True)[0]
ar1['Destination']=ar1.Route.str.split('to',expand=True)[1]
del ar1['Route']
del ar1['Date Flown']
new_index=['Airline Name', 'Overall_Rating', 'Verified', 'Type Of Traveller',
       'Seat Type','Origin','Destination','Month Flown','Year Flown', 'Seat Comfort', 'Cabin Staff Service',
       'Food & Beverages', 'Ground Service', 'Recommended']
ar1=ar1.reindex(columns=new_index)
airline_unique=ar1['Airline Name'].unique()
import json
file_name="airline_unique.json"
unique_airlines_list = airline_unique.tolist()
with open(file_name, 'w') as json_file:
    json.dump(unique_airlines_list, json_file)
from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
le2=LabelEncoder()
le3=LabelEncoder()
le4=LabelEncoder()
le5=LabelEncoder()
le6=LabelEncoder()
le7=LabelEncoder()
le8=LabelEncoder()
le9=LabelEncoder()

# Save LabelEncoders
for i, le in enumerate([le1, le2, le3, le4, le5, le6, le7, le8, le9], start=1):
    with open(f'le{i}.pkl', 'wb') as le_file:
        pickle.dump(le, le_file)

ar1['Airline Name']=le1.fit_transform(ar1['Airline Name'])
ar1['Verified']=le2.fit_transform(ar1['Verified'])
ar1['Type Of Traveller']=le3.fit_transform(ar1['Type Of Traveller'])
ar1['Seat Type']=le4.fit_transform(ar1['Seat Type'])
ar1['Origin']=le5.fit_transform(ar1['Origin'])
ar1['Destination']=le6.fit_transform(ar1['Destination'])
ar1['Month Flown']=le7.fit_transform(ar1['Month Flown'])
ar1['Year Flown']=le8.fit_transform(ar1['Year Flown'])
ar1['Recommended']=le9.fit_transform(ar1['Recommended'])
x=ar1.loc[:,'Airline Name':'Ground Service']
y=ar1['Recommended']
smote=SMOTE(sampling_strategy='auto',random_state=50)
x,y=smote.fit_resample(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)
import pickle
model=KNeighborsClassifier(n_neighbors=10)
model.fit(x_train,y_train)
pred=model.predict(x_test)
pickle.dump(model,open('model.pkl','wb'))
pickle.dump(ss, open('ss.pkl', 'wb'))
