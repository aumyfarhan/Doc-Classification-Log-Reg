# written by farhan
#2018

# import necesssary python mofdules


import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt


# defining chunksize (no of rows) to be read from the training and testing
# csv files at a time
chunk_size=1000


# reading "chunk_size=1000" amount of training example data
# at a time and appending to a list and then concatenating to
# pandas dataframe
chunks=[]

for chunk in pd.read_csv('training.csv', chunksize=chunk_size, header=None,dtype=int):
    chunks.append(chunk)
    
df=pd.concat(chunks,axis=0)

training_all=df.values

# deleteing pandas dataframes to release  memory
del chunks
del chunk
del df

# training data 
X_all=training_all[:,1:-1]
#training data label
label_all=training_all[:,-1]


chunks=[]

# reading "chunk_size=1000" amount of testing data
# at a time and appending to a list and then concatenating to
# pandas dataframe
for chunk in pd.read_csv('testing.csv', chunksize=chunk_size, header=None,dtype=int):
    chunks.append(chunk)
    
df=pd.concat(chunks,axis=0)

testing_all=df.values


# deleteing pandas dataframes to release  memory
del chunks
del chunk
del df

# testing example data
X_test=testing_all[:,1:]
#training example data id
label_id=testing_all[:,0]

# max-min normalization of each example data
X_min=X_test.min(axis=1)
X_min_matrix=np.transpose(np.transpose(np.ones(X_test.shape))*X_min)

X_max=X_test.max(axis=1)
X_max_matrix=np.transpose(np.transpose(np.ones(X_test.shape))*X_max)

X_test=(X_test-X_min_matrix)/(X_max_matrix-X_min_matrix)

# adding extra column of one at the begining
# of each example to be multiplied with bias value 
TT=np.ones((X_test.shape[0],1))
X_test=np.column_stack((TT,X_test))
    

# no of class in the data
class_no=20

# total no of words(considered as feature attributes) in the data
word_no=61188

# defining numpy array weight vectors 
# initially all zero
weight_matrix=np.zeros((class_no,word_no+1)) 

# Defining hyperparameters for the logistic regression training

# learning rate eta
eta=0.01

# regularization factor lambda
Lambda=0.01

# batch size is the number of training example used for
# weight updata at each iteration
batch_size=500

# list to store prediction error made during each iteration
error_val=[]

# initail garbaze error value
old_error=10000

# max no of iteration over all the the training data
max_iter=50
iteration=1

# iterating over all training data one at
for pp in range(0,max_iter):
    
    # iterating through the training data using batch_size amount data 
    # at a time for weight update
    for i in range(0,int(len(X_all)/batch_size)):
    
        # extracting batch_size amount of data
        X_train=X_all[i*batch_size:(i+1)*batch_size,:]
        Y_train=label_all[i*batch_size:(i+1)*batch_size]-1
        
        
    
        # max-min normalization of each example data
        X_min=X_train.min(axis=1)
        X_min_matrix=np.transpose(np.transpose(np.ones(X_train.shape))*X_min)
        
        X_max=X_train.max(axis=1)
        X_max_matrix=np.transpose(np.transpose(np.ones(X_train.shape))*X_max)
        
        X_train=(X_train-X_min_matrix)/(X_max_matrix-X_min_matrix)
        
        # adding extra column of one at the begining
        # of each example to be multiplied with bias value 
        TT=np.ones((X_train.shape[0],1))
        X_train=np.column_stack((TT,X_train))
        
        # calculating the one hot-encoded delta matrix from
        # the current training data label
        delta=np.zeros((class_no,X_train.shape[0]))
        delta[Y_train,np.arange(X_train.shape[0])]=1
        
        # 
        temp_weight_by_input=np.exp(np.matmul(weight_matrix,np.transpose(X_train))) 
        # standardizatipon
        temp_weight_by_input=temp_weight_by_input/(temp_weight_by_input.sum(axis=0)+1)
        
        temp_diff1=(np.matmul((delta-temp_weight_by_input),X_train)-Lambda*weight_matrix)
        temp_diff=eta*temp_diff1
        
        # weight update amount
        new_error=sum(sum(temp_diff1))
        
        error_val.append(new_error)
        iteration +=1
        
        # condition to check previous weight update amount
        # with current weight update amount
        # if the abs difference is less than 0.0001
        # stop the weight update process
        
        if abs((old_error-new_error)/old_error)<0.0001:
            break
        
        old_error=new_error
        weight_matrix=weight_matrix+temp_diff
        
        
        # print the weight update amount after each 10th 
        # iteration
        if (iteration%10 == 0): 
                        # Print info
            print("Iteration: {:d}".format(iteration),
                  "Training lerror: {:10f}".format(new_error))


        

# plotting the weight update amount vs iteration no 
t = np.arange(0,len(error_val))
new_error=error_val/max(error_val)
plt.figure(figsize = (6,6))
plt.plot(t, np.array(new_error))
plt.show()


test_label=[]

# predicting class label of test data example
temp_matrix=np.exp(np.matmul(weight_matrix,np.transpose(X_test)))


temp_label=temp_matrix.argmax(axis=0)+1
test_label=test_label+temp_label.tolist()

   
output_file='log_reg_eta_'+str(eta)+'_lambda_'+str(Lambda)+'_result.csv'

# appending the test data sample id and predicted data sample class label in a list
table=[]
pd=str('id')
cc=str('class')
table.append([pd,cc]) 
for i,j in enumerate(label_id):
    y=test_label[i]
    table.append([j,y])


# creating a csv file of data sample id and their corresponding  class label 
with open(output_file, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in table:
        writer.writerow(val)
