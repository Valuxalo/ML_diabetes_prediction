import os
import pickle

def save_model(model, name):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(f'{root_dir}/artifacts/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
