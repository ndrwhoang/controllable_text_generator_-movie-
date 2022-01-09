import os
import json
import configparser
import pandas as pd
from sklearn.model_selection import train_test_split

def train_val_test_split(data_path, seed):
    data = pd.read_csv(data_path, delimiter=',')
    print(data.columns)
    data = data[['title', 'plot_synopsis', 'tags']]
    train_dev, test = train_test_split(data, test_size=0.1, random_state=seed)
    train, dev = train_test_split(train_dev, test_size=0.1, random_state=seed)
    
    train_out = train.to_dict(orient='records')
    train_subset = train_out[:30]
    dev_out = dev.to_dict(orient='records')
    test_out = test.to_dict(orient='records')
        
    with open(os.path.join('data', 'processed', 'train.json'), 'w') as f:
        json.dump(train_out, f)
    f.close()
    
    with open(os.path.join('data', 'processed', 'train_subset.json'), 'w') as f:
        json.dump(train_subset, f)
    f.close()
    
    with open(os.path.join('data', 'processed', 'dev.json'), 'w') as f:
        json.dump(dev_out, f)
    f.close()
    
    with open(os.path.join('data', 'processed', 'test.json'), 'w') as f:
        json.dump(test_out, f)
    f.close()
    
if __name__ == '__main__':
    config_path = os.path.join('configs', 'config.cfg')
    config = configparser.ConfigParser()
    config.read(config_path)
    train_val_test_split(config['data_path']['data_original'],
                         int(config['general']['seed']))