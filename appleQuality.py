import pandas as pd
import torch
#import torch.nn as nn
#import torchvision.transforms as transforms
'''
#5.	Create your neural network classifier that has some hidden layers
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3: 100 --> 100
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        self.relu3 = nn.ReLU()

        # Linear function 4 (readout): 100 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.relu3(out)

        # Linear function 4 (readout)
        out = self.fc4(out)
        return out

'''
def main():
    #1.	Import the data, you should use some type of pandas api call to read in a csv file to a dataframe
    df=pd.read_csv("apple_quality.csv")


    # 2.Provide some information about your data (exploratory data analysis: EDA), 
    # there are several pandas functions you can use to do this. 
    print(df.head())
    #df.info()

    # Normalize or standardize your values such that you are not dealing with various ranges of values for your features
    for column in df: 
        if(column!="Quality" and column!="A_id"):
            df[column] = pd.to_numeric(df[column])
            #print(type(df[column]))
            df[column] = df[column]  / df[column].abs().max() 
    df["Quality"].replace({"bad": 0, "good": 1}, inplace=True)

    print(df.head())

    #3.	Preprocess your data, prepare you input, output vectors / matrices.
    train=df.sample(frac=0.8,random_state=200)
    test=df.drop(train.index)

