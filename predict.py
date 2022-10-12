import pandas as pd

from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np





def predict(model,x):
    pred = model.predict(x)
    pred = np.array([np.argmax(pred[i]) for i in range(len(pred))])
    return pred

def plot_actual_vs_predicted(y_true,y_pred,title):
    cm = confusion_matrix(y_true,y_pred)
    cm=cm.astype(np.double)
    cm=(np.round(cm / cm.sum(axis=1),2))*100
    #print(cm)
    
    plt.figure(figsize=(5,5))
    index=['Covid','Lung Opacity','Normal','Pneumonia']#['Covid', 'Normal']
    sns.heatmap(cm,cmap="cividis",annot=True,fmt='g',
                xticklabels=index,yticklabels=index)
    plt.title(title)
    plt.show()
    print("Classification Report")
    print(classification_report(y_true,y_pred))


def Metric_Sensivity(y_true,y_pred):
    res = []
    for l in [0,1]:
        prec,recall,_,_ = precision_recall_fscore_support(np.array(y_true)==l,
                                                          np.array(y_pred)==l,
                                                          pos_label=True,average=None)
        res.append([l,recall[0],recall[1]])

    df1=pd.DataFrame(res,columns = ['class','sensitivity','specificity'])
    df1.describe()

    #metrics_auc=roc_auc_score(y_true,y_pred,multi_class='ovr')#model.predict(x_test)
    #print('Metric AUCCCC={:0.4f}'.format(metrics_auc))

    sensitivity,specificity=df1['sensitivity'],df1['specificity']

    print('Sensivity={:0.4f}'.format(sensitivity.mean()))
    print('specificity={:0.4f}'.format(specificity.mean()))
 

def Metric_auc(y_true,y_pred):

    ## Metric Accuracy için   
    #One-hot encoder
    y_valid=y_true.reshape(-1,1)
    ypred=y_pred.reshape(-1,1)
    y_valid = pd.DataFrame(y_valid)
    ypred=pd.DataFrame(ypred)

    onehotencoder = OneHotEncoder()
    y_valid= onehotencoder.fit_transform(y_valid).toarray()
    ypred = onehotencoder.fit_transform(ypred).toarray()
    metrics_auc=roc_auc_score(y_valid,ypred,multi_class='ovr')#model.predict(x_test)
    return metrics_auc 


def Rocc_Curve(y_true,y_pred,metrics_auc):
  #One-hot encoder
  y_valid=y_true.reshape(-1,1)
  ypred=y_pred.reshape(-1,1)
  y_valid = pd.DataFrame(y_valid)
  ypred=pd.DataFrame(ypred)

  onehotencoder = OneHotEncoder()
  y_valid= onehotencoder.fit_transform(y_valid).toarray()
  ypred = onehotencoder.fit_transform(ypred).toarray()

  n_classes = ypred.shape[1]

  # Plotting and estimation of FPR, TPR
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  index=['Covid','Lung Opacity','Normal','Pneumonia']#['Covid', 'Normal']
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_valid[:, i], ypred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

  colors = cycle(['blue', 'green', 'red','darkorange','purple','navy','purple'])
  for i, color in zip(range(n_classes), colors):
    pyplot.plot(fpr[i], tpr[i], color=color, lw=1.5, label='{0}(RocCurveArea= {1:0.4f})' ''.format(index[i], roc_auc[i])) #'ROC curve of {0} (area = {1:0.4f})' ''.format(index[i], roc_auc[i]))

  pyplot.plot([0, 0.95],[0, 0.05], color='navy', lw=1.5, label=" AUC ={:0.2f}%".format(100. *metrics_auc) )  
  pyplot.plot([0, 1], [0, 1], 'k--', lw=1.5)
  pyplot.xlim([-0.05, 1.0])
  pyplot.ylim([0.0, 1.05])
  pyplot.xlabel('False Positive Rate',fontsize=10, fontweight='bold')
  pyplot.ylabel('True Positive Rate',fontsize=10, fontweight='bold')
  pyplot.tick_params(labelsize=12)
  pyplot.legend(loc="lower right")
  ax = pyplot.axes()                              
  pyplot.show()   
    