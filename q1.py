import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class KNNClassifier:
    def __init__(self):
        self.dataset=[]
        self.neighbor=1
    def train(self,path):
        dataset=pd.read_csv(path,header=None)
        self.dataset=dataset
        train_features=dataset.iloc[:,1:].to_numpy()
        train_labels=dataset.iloc[:,0:1].to_numpy()
        
        from sklearn.model_selection import train_test_split
        train_data,valid_data,train_label,valid_label=train_test_split(train_features,train_labels,test_size=0.2,train_size=0.8,shuffle=False)
        
        for neighbor in range(1,4,2):
            valid_pred=[]
            aux=0
            for i in range(len(valid_data)):
                lst=[]
                for j in range (len(train_data)):
                    valid_row=valid_data[i]
                    train_row=train_data[j]
                    d = np.linalg.norm(valid_row-train_row)
                    l=train_label[j][0]
                    lst.append((d,l))
                    lst.sort(key=lambda x: x[0])
                    if len(lst)>neighbor:
                        lst=lst[:-1]
                temp=[]
                for k in range(0,neighbor):
                    temp.append(lst[k][1])
                ans=(np.argmax(np.bincount(temp)))
                # print (ans)
                valid_pred.append(ans)    
            x=accuracy_score(valid_label,valid_pred)
            # print(x)
            if x>aux :
                aux=x                
                self.neighbor=neighbor
            
    def predict(self,path):
        test_features=pd.read_csv(path,header=None).to_numpy()
        dataset=self.dataset
        train_features=dataset.iloc[:,1:].to_numpy()
        train_labels=dataset.iloc[:,0:1].to_numpy()
        test_features=pd.read_csv(path,header=None).to_numpy()        
        neighbor=self.neighbor
                
        valid_pred=[]
        for i in range(len(test_features)):
            lst=[]
            for j in range (len(train_features)):
                test_row=test_features[i]
                train_row=train_features[j]
                d = np.linalg.norm(test_row-train_row)
                l=train_labels[j][0]
                lst.append((d,l))
                lst.sort(key=lambda x: x[0])
                if len(lst)>neighbor:
                    lst=lst[:-1]
            temp=[]
            for k in range(0,neighbor):
                temp.append(lst[k][1])
            ans=(np.argmax(np.bincount(temp)))
            #print (ans)
            valid_pred.append(ans)    
        return valid_pred
    
# obj=KNNClassifier()
# obj.train('train.csv')
# predictions=obj.predict('test.csv')
# test_label=pd.read_csv('test_labels.csv',header=None)

# x=accuracy_score(test_label,predictions)
#print(x)