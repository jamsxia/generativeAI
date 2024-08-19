import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torchvision.transforms as transforms

#5.	Create your neural network classifier that has some hidden layers
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, 128)
        # Non-linearity 2
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        # Linear function 3: 100 --> 100
        self.fc3 = nn.Linear(128, 64)
        # Non-linearity 3
        self.relu3 = nn.ReLU()

        # Linear function 4 (readout): 100 --> 10
        self.fc4 = nn.Linear(64, output_dim)  
        self.softmax = nn.Softmax()


    def forward(self, x):
        # Linear function 1
        x = self.fc1(x)
        # Non-linearity 1
        x = self.relu1(x)

        x = self.dropout(x)

        # Linear function 2
        x = self.fc2(x)
        # Non-linearity 2
        x = self.relu2(x)

        # Linear function 2
        x = self.fc3(x)
        # Non-linearity 2
        x = self.relu3(x)

        # Linear function 4 (readout)
        x = self.fc4(x)
        return x


def main():
    #1.	Import the data, you should use some type of pandas api call to read in a csv file to a dataframe
    df=pd.read_csv("apple_quality.csv")
    label_encoder = LabelEncoder()



    # 2.Provide some information about your data (exploratory data analysis: EDA), 
    # there are several pandas functions you can use to do this. 
    print(df.head())
    df.info()

    # Normalize or standardize your values such that you are not dealing with various ranges of values for your features
    labels_array = df['Quality'].to_numpy()
    label_encoder.fit(labels_array)
    labels_array = label_encoder.fit_transform(labels_array)
    data_array = df.loc[:, 'Size':'Ripeness']
    data_array = (data_array + data_array.min().abs()) / (data_array.max() + data_array.min().abs())
    data_array=data_array.to_numpy()
    train_input, test_input, train_output, test_output = train_test_split(data_array, labels_array,
                                                    test_size=0.1, random_state=42)
    train_input, test_input = torch.from_numpy(train_input).float(), torch.from_numpy(test_input).float()


    '''for column in df: 
        if(column!="Quality" and column!="A_id"):
            df[column] = pd.to_numeric(df[column])
            #print(type(df[column]))
            df[column] = df[column]  / df[column].abs().max() 
    df["Quality"].replace({"bad": 0, "good": 1}, inplace=True)'''

    print(df.head())

    #3.	Preprocess your data, prepare you input, output vectors / matrices.

    '''train=df.sample(frac=0.8,random_state=200)
    test=df.drop(train.index)
    #train=df.iloc[train_index]
    train_input=train.loc[:, (df.columns != 'Quality')& (df.columns!="A_id")].astype("float32")
    train_output=train['Quality'].astype("float32")
    train_input=torch.tensor(train_input.values)
    train_output=torch.tensor(train_output.values)
    

    test_input=test.loc[:, (df.columns != 'Quality')& (df.columns!="A_id")].astype("float32")
    test_output=test['Quality'].astype("float32")
    test_input=torch.tensor(test_input.values)
    test_output=torch.tensor(test_output.values)'''
    #print(test_input)

    #6.Use CrossEntropyLoss and the Adam optimizer to train your Neural network.
    data_shape=data_array.shape[1]
    #label_shape=labels_array.shape[1]
    fnn=FeedforwardNeuralNetModel(data_shape,256,2)
    #criterion = nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss() 
    rate_learning=1e-4
    optim = torch.optim.Adam(fnn.parameters(), lr=rate_learning)

    ## trainning, ideally need to be in a separate file, but it's simple model, so good enough, I guess(8/11/2024)
    for epoch in range(200):
        running_loss=0
        i=0
        for input, label in zip(train_input, train_output): 
            i+=1
            
            optim.zero_grad()
            output=fnn(input).data
            #label=label.unsqueeze(1)
            label=label.reshape(1)
            #print(output.shape)
            #print(label.shape)
            #print(output,label)
            #print(output.dtype)
            #print(label.dtype)
            #print(output,label)
            loss=criterion((output+1)/2,label)
            loss.requires_grad = True
            loss.backward()
            optim.step()

            running_loss+=loss.item()
            if i % 100 == 0:    
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print("Finished Training")

    ## testing 
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for input, label in zip(test_input, test_output): 
            output = fnn(input)
            # the class with the highest energy is what we choose as prediction
            predicted=1 if (output.data.item()>0) else 0
            label=label.item()
            total+=1
            correct += (predicted == label)

##report accuracy
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

main()
