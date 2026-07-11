from load_data import load_data
from train import train
from predict import predict
from save_model import save_model
from processing import processor, split_data
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class MLPipeline:
    def __init__(self):
        self.model = None
        self.data = None
        self.train = None
        self.val = None
        self.test = None
        self.name_model = os.getenv("MODEL_NAME")
        
    def run_full_pipeline(self):
            self.data = load_data(load_path=None)
            if not self.data.empty:
                self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.data)
                process = processor()
                self.model = train(process, self.X_train, self.y_train)
                if self.model is not None:
                    predict(self.model, self.X_test, self.y_test)
                save_model(model=self.model, name=self.name_model)
                print("Pipeline выполнен успешно!")
    

# Использование одной командой
if __name__ == "__main__":
    pipeline = MLPipeline()
    model = pipeline.run_full_pipeline()