import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df=pd.read_csv("train.csv")
x_train,x_test,y_train,y_test=train_test_split(df.dead(0) or df.alive(1),df.compansation)
lgreg=LogisticRegression()
lgreg.fit(x_train,y_train)
lgreg.preet(x_test[0:10])
pre=lgreg.predict(x_test)
score=lgreg.score(x_test,y_test)
print(score)
