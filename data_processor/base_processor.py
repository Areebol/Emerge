import os

class BaseProcessor:
    def __init__(self, model, tokenizer, model_config) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.max_sample = 50
        self.exp_dir = "./exp"
        
    def process(self, index, data, model_generate):
        if index > self.max_sample:
            return False
        processed_data = self.process_data(index, data, model_generate)
        self.save_data(index, processed_data)
        return True
        
    def process_data(self, data, model_generate):
        raise NotImplementedError("Subclasses should implement this method.")

    def save_data(self, data):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def set(self):
        os.makedirs(self.save_dir,exist_ok=True) 
        self.save_file = f"{self.save_dir}/{self.model_config[0]}.csv"
        if os.path.exists(self.save_file):
            os.remove(self.save_file)
    
    def reset(self):
        raise NotImplementedError("Subclasses should implement this method.")
        