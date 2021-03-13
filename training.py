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

def train(df, data_train, data_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # lstm_train_model = LSTM()
    model = LSTM().to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 3
    timestamp = 5
    test_size = 30
    best_acc = 0
    for e, i in tqdm(enumerate(range(epochs)), total = epochs):
        model.train()
        total_loss, total_acc = [], []
        date_ori = pd.to_datetime(data_train.iloc[:, 0]).tolist()
        for k in range(0, data_train.shape[0] - 1, timestamp):
            index = min(k + timestamp, data_train.shape[0] - 1)
            batch_x = np.expand_dims(data_train.iloc[k : index, :].values, axis = 0)
            batch_y = data_train.iloc[k + 1 : index + 1, :].values
            batch_x = torch.Tensor(batch_x).to(device)
            batch_y = torch.Tensor(batch_y).to(device)
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                            torch.zeros(1, 1, model.hidden_layer_size).to(device))
            logits = model(batch_x)
            single_loss = loss_function(logits, batch_y)
            single_loss.backward()
            optimizer.step()
            total_acc.append(calculate_accuracy(batch_y[:, 0].cpu().detach().numpy(),logits.cpu().detach().numpy()))
            total_loss.append(single_loss.item())
        loss_epoch = np.sum(np.array(total_loss))/len(total_loss)
        acc_epoch = np.sum(np.array(total_acc))/len(total_acc)
            
        model.eval()
        test_size = len(data_test)
        future_day = test_size
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
        if i%1 ==0:
            print("Epoch: {}, acc: {}, loss: {}".format(i, acc_epoch,loss_epoch))
            os.makedirs("weights", exist_ok = True)
            epoch_eval_acc = calculate_accuracy(output_predict,df.values)
            print("acc_eval: ",epoch_eval_acc )
            if epoch_eval_acc > best_acc:
                best_acc = epoch_eval_acc
                torch.save(model.state_dict(), "weights/checkpont_{}.pth".format(epoch_eval_acc))
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
    df1 = df[(start):]
    train(df, data_train, data_test)