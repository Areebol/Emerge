class BaseLoader:
    def load_data(self,len = 200):
        raise NotImplementedError("Subclasses should implement this method.")
