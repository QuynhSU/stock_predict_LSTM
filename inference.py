from model import LSTM
from eval import eval, calculate_accuracy
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
import os
warnings.filterwarnings("ignore")
def infer(minmax,data_train,data_test ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # lstm_train_model = LSTM()
    model = LSTM().to(device)
    model.load_state_dict(torch.load("D:\stock\weights\checkpont_67.27376310428824.pth"))
    model.eval()
    test_size = len(data_test)
    future_day = test_size
    timestamp = 5
    output_predict = np.zeros((data_train.shape[0] + future_day, data_train.shape[1]))
    output_predict[0] = data_train.iloc[0]
    for k in range(0, (data_train.shape[0] // timestamp) * timestamp, timestamp):
        index = min(k + timestamp, output_predict.shape[0] - 1)
        batch_x = np.expand_dims(df.iloc[k : index, :].values, axis = 0)
        batch_y = df.iloc[k + 1 : index + 1, :].values
        batch_x = torch.Tensor(batch_x).to(device)
        batch_y = torch.Tensor(batch_y).to(device)
        out_logits = model(batch_x)
        # init_value = last_state
        output_predict[k + 1 : k + timestamp + 1] = out_logits.cpu().detach().numpy()[0]
    output_predict = minmax.inverse_transform(output_predict)
    return output_predict
    
if __name__ == "__main__":
    df = pd.read_csv("BID.csv").drop("original",1)
    pro = 0.3
    minmax = MinMaxScaler().fit(df[['Close']].astype('float32')) # Close index
    df_log = minmax.transform(df[['Close']].astype('float32')) # Close index
    df_log = pd.DataFrame(df_log)
    df = df_log
    no_test = 30
    no_train = len(df) - no_test
    start = no_train%5

    data_train  = df[(start-1):no_train]
    data_test = df[no_train:]
    # print(no_train, no_test)
    df1 = df[(start):]
    results= infer(minmax,data_train,data_test ) 
    print(results[:-30])