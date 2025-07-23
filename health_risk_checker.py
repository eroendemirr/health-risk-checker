import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df=pd.read_csv("patient_risk_data.csv")

X=df.drop("label",axis=1)
y=df["label"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)

rf_model=RandomForestClassifier(n_estimators=100,random_state=0)
rf_model.fit(X_train,y_train)

y_pred=rf_model.predict(X_test)

print(classification_report(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred,labels=rf_model.classes_)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=rf_model.classes_)
disp.plot(cmap=plt.cm.Greens)
plt.show()