import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from metrics_visualization import *
import numpy as np
from Load_Future_Vektor import create_Vektor
from CovidDWNet import CovidDWNet



def models(x, y,xtest,ytest):
    
    accuracy = []
    f1score = []
    model = []   
    model.append(GradientBoostingClassifier(random_state=101))
    
    for i in model:
        mdl = i
        i.fit(x, y)
        pred = i.predict(xtest)
        #print(pred)
        accuracy.append((round(accuracy_score(ytest, pred), 2))*100)
        #f1score.append((round(f1_score(ytest, pred), 2))*100)        
        print(f'Model: {i}\nAccuracy: {accuracy_score(ytest, pred)}\n\n')
        
        plot_actual_vs_predicted(ytest,pred,"Test Data Predictions")
        #grafik(pred)    
        Metric_Sensivity(ytest,pred) 
        metrics_auc=Metric_auc(ytest,pred) 
        print('Metric AUC={:0.4f}'.format(metrics_auc))
        print('')

        Rocc_Curve(ytest,pred,metrics_auc)
                                 

def test(checkpoint_path,data_path):
    
    model = CovidDWNet(inpt_shape = (128, 128, 3), num_class = 4)
    
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    latest = checkpoint_path# tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    
    v_X_train, v_X_test, v_y_train, v_y_test=create_Vektor(model=model,data_path=data_path)
    v_y_test = np.argmax(v_y_test, axis=1)
    v_y_train = np.argmax(v_y_train, axis=1)
    models(v_X_train,v_y_train,v_X_test,v_y_test)



if __name__ == '__main__':
    checkpoint_path = "checkpoint/our_model.h5"
    data_path="data"
    test(checkpoint_path,data_path)