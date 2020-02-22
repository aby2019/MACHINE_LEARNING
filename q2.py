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
        #self.dataset=dataset

        from sklearn import preprocessing
        label_encoder=preprocessing.LabelEncoder()
        dataset[0]=label_encoder.fit_transform(dataset[0])
        self.dataset=dataset
        train_features=dataset.iloc[:,1:].to_numpy()
        train_label_encoded=dataset.iloc[:,0:1].to_numpy()
        attributes=[]
        attributes.append(['b','c','x','f','k','s'])
        attributes.append(['f','g','y','s'])
        attributes.append(['n','b','c','g','r','p','u','e','w','y'])
        attributes.append(['t','f'])
        attributes.append(['a','l','c','y','f','m','n','p','s'])
        attributes.append(['a','d','f','n'])
        attributes.append(['c','w','d'])
        attributes.append(['b','n'])
        attributes.append(['k','n','b','h','g','r','o','p','u','e','w','y'])
        attributes.append(['e','t'])
        attributes.append(['b','c','u','e','z','r'])#
        attributes.append(['f','y','k','s'])
        attributes.append(['f','y','k','s'])
        attributes.append(['n','b','c','g','o','p','e','w','y'])
        attributes.append(['n','b','c','g','o','p','e','w','y'])
        attributes.append(['p','u'])
        attributes.append(['n','o','w','y'])
        attributes.append(['n','o','t'])
        attributes.append(['c','e','f','l','n','p','s','z'])
        attributes.append(['y','w','u','o','r','h','b','n','k'])
        attributes.append(['y','v','s','n','c','a'])
        attributes.append(['g','l','m','p','u','w','d'])
        
        train_feature_encoded=pd.DataFrame(columns=None)
        
        for i in range(1,23):
            dummies=pd.get_dummies(train_features[:,i-1],prefix='',prefix_sep='')
            dummies=dummies.reindex(columns=attributes[i-1]).fillna(0)
            train_feature_encoded=pd.concat([train_feature_encoded,dummies],axis=1,sort=False)
        
        train_feature_encoded=train_feature_encoded.to_numpy()
        from sklearn.model_selection import train_test_split
        train_data,valid_data,train_label,valid_label=train_test_split(train_feature_encoded,train_label_encoded,test_size=0.2,train_size=0.8,shuffle=False)
        
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
                #print (ans)
                valid_pred.append(ans)
            x=accuracy_score(valid_label,valid_pred)
            if x>aux :
                aux=x                
                self.neighbor=neighbor
            
    def predict(self,path):
        dataset=self.dataset
        attributes=[]
        attributes.append(['b','c','x','f','k','s'])
        attributes.append(['f','g','y','s'])
        attributes.append(['n','b','c','g','r','p','u','e','w','y'])
        attributes.append(['t','f'])
        attributes.append(['a','l','c','y','f','m','n','p','s'])
        attributes.append(['a','d','f','n'])
        attributes.append(['c','w','d'])
        attributes.append(['b','n'])
        attributes.append(['k','n','b','h','g','r','o','p','u','e','w','y'])
        attributes.append(['e','t'])
        attributes.append(['b','c','u','e','z','r'])#
        attributes.append(['f','y','k','s'])
        attributes.append(['f','y','k','s'])
        attributes.append(['n','b','c','g','o','p','e','w','y'])
        attributes.append(['n','b','c','g','o','p','e','w','y'])
        attributes.append(['p','u'])
        attributes.append(['n','o','w','y'])
        attributes.append(['n','o','t'])
        attributes.append(['c','e','f','l','n','p','s','z'])
        attributes.append(['y','w','u','o','r','h','b','n','k'])
        attributes.append(['y','v','s','n','c','a'])
        attributes.append(['g','l','m','p','u','w','d'])
        
        test_features=pd.read_csv(path,header=None).to_numpy()       
        train_features=dataset.iloc[:,1:].to_numpy()
        train_labels=dataset.iloc[:,0:1].to_numpy()
        
        train_feature_encoded=pd.DataFrame(columns=None)

        for i in range(1,23):
            dummies=pd.get_dummies(train_features[:,i-1],prefix='',prefix_sep='')
            dummies=dummies.reindex(columns=attributes[i-1]).fillna(0)
            train_feature_encoded=pd.concat([train_feature_encoded,dummies],axis=1,sort=False)

        train_feature_encoded=train_feature_encoded.to_numpy()

        test_feature_encoded=pd.DataFrame(columns=None)
        
        
        for i in range(1,23):
            dummies=pd.get_dummies(test_features[:,i-1],prefix='',prefix_sep='')
            dummies=dummies.reindex(columns=attributes[i-1]).fillna(0)
            test_feature_encoded=pd.concat([test_feature_encoded,dummies],axis=1,sort=False)
        
        
        neighbor=self.neighbor
        test_features=test_feature_encoded.to_numpy()       
        train_features=train_feature_encoded            
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
            if ans==0:
                valid_pred.append('e')
            else:
                valid_pred.append('p')
        return valid_pred
    
# obj=KnnClassifier()
# obj.train('train.csv')
# predictions=obj.predict('test.csv')
# test_label=pd.read_csv('test_labels.csv',header=None)
# x=accuracy_score(test_label,predictions)
# print(x)





#####
#dataset=pd.read_csv('train.csv',header=None)
#attributes=[]
#attributes.append(['b','c','x','f','k','s'])
#attributes.append(['f','g','y','s'])
#attributes.append(['n','b','c','g','r','p','u','e','w','y'])
#attributes.append(['t','f'])
#attributes.append(['a','l','c','y','f','m','n','p','s'])
#attributes.append(['a','d','f','n'])
#attributes.append(['c','w','d'])
#attributes.append(['b','n'])
#attributes.append(['k','n','b','h','g','r','o','p','u','e','w','y'])
#attributes.append(['e','t'])
#attributes.append(['b','c','u','e','z','r'])#
#attributes.append(['f','y','k','s'])
#attributes.append(['f','y','k','s'])
#attributes.append(['n','b','c','g','o','p','e','w','y'])
#attributes.append(['n','b','c','g','o','p','e','w','y'])
#attributes.append(['p','u'])
#attributes.append(['n','o','w','y'])
#attributes.append(['n','o','t'])
#attributes.append(['c','e','f','l','n','p','s','z'])
#attributes.append(['y','w','u','o','r','h','b','n','k'])
#attributes.append(['y','v','s','n','c','a'])
#attributes.append(['g','l','m','p','u','w','d'])
#
#train_feature_encoded=pd.DataFrame(columns=None)
#
#for i in range(1,23):
#    dummies=pd.get_dummies(train_features[:,i-1],prefix='',prefix_sep='')
#    dummies=dummies.reindex(columns=attributes[i-1]).fillna(0)
#    train_feature_encoded=pd.concat([train_feature_encoded,dummies],axis=1,sort=False)
#test_features=pd.read_csv('test.csv',header=None).to_numpy()
#train_features=dataset.iloc[:,1:].to_numpy()
#train_labels=dataset.iloc[:,0:1].to_numpy()
#test_feature_encoded=pd.DataFrame(columns=None)
#
#for i in range(1,23):
#    dummies=pd.get_dummies(test_features[:,i-1],prefix='',prefix_sep='')
#    dummies=dummies.reindex(columns=attributes[i-1]).fillna(0)
#    test_feature_encoded=pd.concat([test_feature_encoded,dummies],axis=1,sort=False)
#
#neighbor=3
#test_features=test_feature_encoded.to_numpy()                
#valid_pred=[]
#for i in range(len(test_features)):
#    lst=[]
#    for j in range (len(train_features)):
#        test_row=test_features[i]
#        train_row=train_features[j]
#        d = np.linalg.norm(test_row-train_row)
#        l=train_labels[j][0]
#        lst.append((d,l))
#        lst.sort(key=lambda x: x[0])
#        if len(lst)>neighbor:
#            lst=lst[:-1]
#    temp=[]
#    for k in range(0,neighbor):
#        temp.append(lst[k][1])
#    ans=(np.argmax(np.bincount(temp)))
#    if ans==0:
#        valid_pred.append('e')
#    else:
#        valid_pred.append('p')
#
#
#
#test_label=pd.read_csv('test_labels.csv',header=None)
#x=accuracy_score(test_label,valid_pred)
#print(x)
##
#
