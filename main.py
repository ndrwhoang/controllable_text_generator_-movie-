import os
import configparser

from src.interface import Interface

if __name__ == '__main__':
    print('hello world')
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    interface = Interface(config)
    interface.run_trial_training('test_run')
    
