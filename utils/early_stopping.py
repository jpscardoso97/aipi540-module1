import torch

class EarlyStopping:
    def __init__(self, patience=5, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_metric = float('inf')  # Initialize with a large value for minimization
        self.best_state_dict = None

    def __call__(self, metric, model):
        if self.restore_best_weights and self.best_state_dict is None:
            self.best_state_dict = model.state_dict()

        if metric > self.best_metric:  
            self.counter += 1
            if self.counter >= self.patience:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                return True  # Early stopping triggered
        else:
            self.best_metric = metric
            self.counter = 0
            if self.restore_best_weights:
                self.best_state_dict = model.state_dict()

        return False  # Continue training

    def restore_best_model(self, model):
        if self.restore_best_weights and self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
