import copy
import os

from matplotlib import pyplot as plt
import ray.tune
import ray.train
import torch

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

class PlotMetricsCallback(ray.tune.Callback):
    def __init__(self, allow_list=None):
        self.metric_data = {}
        self.trial_dir = None
        if allow_list is None:
            allow_list = ['time_this_iter_s', 'gamma', 'lambda', 
                          'learning_rate', 'val_overall_loss', 'val_y_APS',
                          'val_y_accuracy', 'val_y_auc']
        self.allow_list = allow_list

    def on_trial_start(self, iteration, trials, trial, **info):
        self.trial_dir = trial.local_path
        self.trial_name = trial.trial_id  # Use trial_id as the trial name

    def on_trial_result(self, iteration, trials, trial, result):
        if not self.metric_data:
            self.metric_data = {metric_name: [] for metric_name in result.keys() if metric_name in self.allow_list}

        for metric_name, metric_value in result.items():
            if metric_name in self.allow_list:
                self.metric_data[metric_name].append(metric_value)

        for metric_name, metric_values in self.metric_data.items():
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(1, len(metric_values) + 1), metric_values, label=metric_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(self.trial_dir, f"progress_{metric_name}_{self.trial_name}.png"))
            plt.close(fig)

    def on_trial_complete(self, iteration, trials, trial, **info):
        self.metric_data = {}