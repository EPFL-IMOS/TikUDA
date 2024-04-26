import os 

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler

    
def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def scale_feature(x_train, y_train, x_test, y_test, preporcessing = 'min_max'):
    if(preporcessing == 'min_max'):
        scaler = MinMaxScaler()
    elif(preporcessing ==  'std'):
        scaler = StandardScaler()
    else:
        print('Preprocessing Function not implemented')
     
    x_train = scaler.fit_transform(x_train)
    x_test   = scaler.transform(x_test)
    
    y_train = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test   = scaler.transform(y_test.reshape(-1, 1))
    
    return x_train, y_train, x_test, y_test, scaler
    

def training_testing_dataframes(source_file,target_file,window_size = 16):
    #Read data
    df_source = pd.read_csv(source_file, sep = ';')
    df_target = pd.read_csv(target_file, sep = ';')
    
    df_source = df_source.drop('date', axis = 1).values
    df_target = df_target.drop('date', axis = 1).values

    x_source, y_source, x_target, y_target = df_source[:,1:] , df_source[:,0], df_target[:,1:] , df_target[:,0]
    
    #Scale data
    x_source, y_source, x_target, y_target, y_scaler = scale_feature(x_source, y_source, x_target, y_target)
    
    #Convert Dataframe into time sequence
    x_source = create_sequences(x_source,window_size)
    y_source = create_sequences(y_source,window_size)
    
    x_target = create_sequences(x_target,window_size)
    y_target = create_sequences(y_target,window_size)
    
    return x_source, y_source, x_target, y_target, y_scaler

def seed_everything(seed = 42): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True