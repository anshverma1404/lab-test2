from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split



data=load_diabetes()
data
X=data.data
Y=data.target

plt.plot(X,Y)
plt.show()
print(data.feature_names)

 

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
plt.colorbar(X,X_train)
plt.show()
df=pd.DataFrame(data.data,data.feature_names)
df['target']=data.target
