import copy

import torch
import ray.tune
import ray.train

class SaveBestModelCallback(ray.tune.Callback):
    def __init__(self, dataloaders_dict, best_model_path="best_model.pth"):
        super().__init__()
        self.dataloaders_dict = dataloaders_dict
        self.best_model_path = best_model_path
        self.best_val_loss = float('inf')
        self.best_model_state = None  # Placeholder for the best model's state

    def handle_result(self, results, **info):
        val_loss = results["val_loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            # Save the best model state
            self.best_model_state = copy.deepcopy(info["model_state"])
            torch.save(self.best_model_state, self.best_model_path)

    def on_training_end(self, logs=None):
        # Load the best model and evaluate on the test set
        if self.best_model_state is not None:
            model = self.trainer.get_model()
            model.load_state_dict(torch.load(self.best_model_path))
            test_loss = model.validate(
                self.dataloaders_dict['test'], 
                model=model.gp_model, 
                loss_fn=model.loss_fn
            )
            print(f"Final Test Loss: {test_loss}")
            ray.train.report(test_loss=test_loss)
