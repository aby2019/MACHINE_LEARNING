import numpy as np
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt

class DecisionTree:
    
    def __init__(self):
        self.train_features=[]#dataset
        self.index=[]
        self.lst=[]
        self.del_index=[]
        self.data_type=[]
        self.head=None
        
    class root:
        def __init__(self):
            self.left=None
            self.right=None
            self.attribute=None
            self.mean_value=None
            
    def mse_numerical(self,uni,column,train_labels):    
        mean_value=[]
        mse_value=[]
        if(len(uni)==1):
            mean_value.append(uni[0])
        else:
            
            for i in range(len(uni)-1):
                mean_value.append((uni[i]+uni[i+1])/2)
        
        for k in range(len(mean_value)):
            set1=[]
            set2=[]        
            temp=mean_value[k]
            for i in range(len(column)):
                if column[i]>temp:
                    set1.append(train_labels[i])
                else:
                    set2.append(train_labels[i])
            m1=0
            for i in range(len(set1)):
                m1=m1+set1[i]
            if(len(set1)==0):
                m1=1
            else:
                m1=m1/len(set1)
            
            m2=0
            for i in range(len(set2)):
                m2=m2+set2[i]
            if(len(set2)==0):
                m2=1
            else:
                m2=m2/len(set2)        
            mse1=0
            for i in range(len(set1)):##
                mse1=mse1+np.square(np.subtract(set1[i],m1))/len(set1)
            
            mse2=0
            for i in range(len(set2)):
                mse2=mse2+np.square(np.subtract(set2[i],m2))/len(set2)    
        
            ans=(mse1*len(set1)+mse2*len(set2))/(len(set1)+len(set2))
            
            mse_value.append(ans)
        
        #return mse_value,mean_value
        return mean_value[np.argmin(mse_value)],mse_value[np.argmin(mse_value)]

    def mse_categorical(self,uni,column,train_labels):  
    
        mse_value=[]
        for k in range(len(uni)):
            temp=uni[k]
            set1=[]
            set2=[]
            for i in range(len(column)):
                if column[i]==temp:
                    set1.append(train_labels[i])
                else:
                    set2.append(train_labels[i])
            m1=0
            for i in range(len(set1)):
                m1=m1+set1[i]
            if(len(set1)==0):
                m1=1
            else:
                m1=m1/len(set1)
            
            m2=0
            for i in range(len(set2)):
                m2=m2+set2[i]
            if(len(set2)==0):
                m2=1
            else:
                m2=m2/len(set2)        
    
            
            mse1=0
            for i in range(len(set1)):##
                mse1=mse1+np.square(np.subtract(set1[i],m1))
            mse2=0
            for i in range(len(set2)):
                mse2=mse2+np.square(np.subtract(set2[i],m2)) 
            
            ans=(mse1*len(set1)+mse2*len(set2))/(len(set1)+len(set2))
            
            mse_value.append(ans)
            
            #return mse_value,mean_value
        return uni[np.argmin(mse_value)],mse_value[np.argmin(mse_value)]

    def tree_build(self,train_features,cur_level,max_level):

        data_type=self.data_type
        self.head=None
        if(len(train_features)==0):
            return None 
        if (cur_level==max_level):
            return None
            
        if (len(train_features)<10):
            return None
        break_point=[]
        break_mse=[]
        train_labels=train_features[:,-1]
        for i in range(len(train_features[0])-1):
            if data_type[i] == 'object':
                m,attr=self.mse_categorical(np.unique(train_features[:,i]),train_features[:,i],train_labels)
            else:
                m,attr=self.mse_numerical(np.unique(train_features[:,i]),train_features[:,i],train_labels)
            break_point.append(m)
            break_mse.append(attr)
            
        
    #    train_features=pd.DataFrame(train_features)
    #    train_labels=pd.DataFrame(train_labels)
        inx=np.argmin(break_mse)
        mean_break=break_point[inx]
        col_value=inx
            
        node=self.root()
        node.attribute=col_value
        node.mean_value=mean_break
       # print("inx=",inx,mean_break)
        
        
        if(type(train_features[0][col_value])==str):###
            left_frame=train_features[train_features[:,inx]==mean_break]
            right_frame=train_features[train_features[:,inx]!=mean_break]
    #        left_frame=np.delete(left_frame,inx,1)
    #        right_frame=np.delete(right_frame,inx,1)
            
            node.left=self.tree_build(left_frame,cur_level+1,max_level)
            node.right=self.tree_build(right_frame,cur_level+1,max_level)   
        else:
            left_frame=train_features[train_features[:,inx]<=mean_break]
            right_frame=train_features[train_features[:,inx]>mean_break]
            
    #        left_frame=np.delete(left_frame,inx,1)
    #        right_frame=np.delete(right_frame,inx,1)
           # print(len(left_frame)+len(right_frame))
            node.left=self.tree_build(left_frame,cur_level+1,max_level)
            node.right=self.tree_build(right_frame,cur_level+1,max_level)   
        return node
    
    def pred(self,train_features,test_features,head):

        if(head.left==None):
            print ('ans=',train_features[:,-1].mean())
            return (train_features[:,-1].mean())
        if(head.right==None):
            print ('ans=',train_features[:,-1].mean())
            return (train_features[:,-1].mean())
        
        #    train_features=pd.DataFrame(train_features)
    #    train_labels=pd.DataFrame(train_labels)
        mean_break=head.attribute
        mean_value=head.mean_value
        # print("node value=",head.attribute,head.mean_value,end='')
        # print("test value=",test_features[mean_break],"\n")
      
        if(type(test_features[mean_break])==str):###
            left_frame=train_features[train_features[:,mean_break]==mean_value]
            right_frame=train_features[train_features[:,mean_break]!=mean_value]
            if(test_features[mean_break]==mean_value):
                return self.pred(left_frame,test_features,head.left)
            else:
                return self.pred(right_frame,test_features,head.right)   
        else:
            left_frame=train_features[train_features[:,mean_break]<=mean_value]
            right_frame=train_features[train_features[:,mean_break]>mean_value]
            if(test_features[mean_break]<=mean_value):
                return self.pred(left_frame,test_features,head.left)
            else:            
                return self.pred(right_frame,test_features,head.right)
        
    def train(self,path):
        
        dataset=pd.read_csv(path)
        train_features=dataset
        lst=(train_features.isna().sum()/1000)*100
        index=[]
        for i in train_features:
            index.append(i)
        del_index=[]
        for i in range(len(lst)):
            if lst[i]>50:
                del train_features[index[i]]
                del_index.append(index[i])
        
        index=[]
        
        for i in train_features:
            index.append(i)
        lst=(train_features.isna().sum()/1000)*100
        data_type=train_features.dtypes
            
        for i in range(len(index)):
            if data_type[i] == 'object':
                if lst[i]!=0:    
                    train_features[index[i]].fillna(train_features[index[i]].mode()[0],inplace=True)
            else:
                if lst[i]!=0:
                    train_features[index[i]].fillna(train_features[index[i]].mean(),inplace=True,axis=0)
                    #print(train_features[index[i]].mean(),i)
        
        train_features=train_features.to_numpy()
        self.train_features=train_features#dataset
        self.index=index
        self.lst=lst
        self.del_index=del_index
        self.data_type=data_type
        self.head=self.tree_build(train_features,0,10)
    def predict(self,path):
        
        train_features=self.train_features
        test_features=pd.read_csv(path)
        #self.inorder(self.head)
        del_index=self.del_index
        for i in range(len(del_index)):
            del test_features[del_index[i]]
            
        test_features=test_features.to_numpy()
        predictions=[]
        for i in range(len(test_features)):
            aux=self.pred(train_features,test_features[i],self.head)
            predictions.append(aux)
        return predictions
    def inorder(self,head):
        if(head==None):
            return
        #print(head.attribute,head.mean_value)
        self.inorder(head.left)
        print(head.attribute,head.mean_value)
        self.inorder(head.right)
        
#obj=DecisionTree()
#obj.train()

# dtree_regressor = DecisionTree()
# dtree_regressor.train('train.csv')
# #dtree_regressor.inorder(dtree_regressor.head)
# predictions = dtree_regressor.predict('test.csv')
# test_label=pd.read_csv('test_labels.csv',header=None)
# test_label=test_label.iloc[:,1]
                           
# from sklearn.metrics import r2_score
# print(r2_score(test_label,predictions))

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    