import sklearn.base
import time

EPOCH_BLOCKS = 1

class GeometricBaseEstimator(sklearn.base.BaseEstimator):
    def __init__(self):
        raise NotImplementedError("This class is not fully implemented yet.")
    def _initialize_reporting_lists(self):
        self.model_states = []
        self.validation_predictions = {}
        self.validation_psych_preds = {}
        self.train_loss_list = []
        self.val_loss_list = []
        self.start_time = time.time()
        self.alpha_list = []
        self.gamma_list = []
        self.lambda_list = []
        self.beta_list = []
        self.val_auc_list = []
            
    def fit(self, X=None, y=None):
        """
        Fit method for sklearn models.

        NOTE: Since the labels for sklearn can be in the form of 0, 1, 2, 3, etc., 
        we can't have l and y as input. We can only have X and y,
        where y the transformation of l and y to a single array.

        Args: 
            X (array-like): The input features.
            y (array-like): The target values. This should be a single array that is the 
                            result of transforming l and y to a single array.
        """
        l, y = self._separate_labels(y)
        self._check_input(X, l, y)
        self._setup(X, l, y)
        self._initialize_reporting_lists()

        best_val_loss = float('inf')
        best_model_state = None
        for epoch in range(self.num_epochs):
            start_time = time.time()

            train_loss = self._train_one_epoch()
            self.train_loss_list.append(train_loss)

            should_validate = True if "val" in self.dataloaders else None
            if should_validate:
                val_loss, val_auc = self._validate(self.dataloaders["val"], self.gp_model, self.likelihood, self.mll, self.device)
                self.val_loss_list.append(val_loss)
                self.val_auc_list.append(val_auc)

            if epoch % EPOCH_BLOCKS == EPOCH_BLOCKS - 1:
                print(f"Iteration {epoch}: Training loss = {train_loss.item()}, Validation loss = {should_validate}, Validation AUC = {should_validate}. Elapsed time: {time.time() - start_time}s,  beta:{self.likelihood.variational_mean_beta}")#, alpha:{likelihood.variational_mean_alpha},")


            if "val" in self.dataloaders:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save the best model state
                    best_model_state = {
                        "gp": self.gp_model.state_dict(),
                        "likelihood": self.likelihood.state_dict()
                    }
            else:
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    # Save the best model state
                    best_model_state = {
                        "gp": self.gp_model.state_dict(),
                        "likelihood": self.likelihood.state_dict()
                    }

        # Load best model state if available
        if best_model_state:
            self.gp_model.load_state_dict(best_model_state["gp"])
            self.likelihood.load_state_dict(best_model_state["likelihood"])

        self.gp_model.eval()
        self.likelihood.eval()
        return self

    def predict_proba(self, X):
        """
        Predicts the probability of l given X, i.e. p(l|X).
        Note that this is not the same as predict_prob_l_given_y_X. 
        In fact, due to p(y=1|l=1) = 1, 

                    p(l|X) = p(l|y=1, X) * p(y=1|X)

        Parameters:
            X (array-like): The input data to predict probabilities for.

        Returns:
            array-like: The predicted probability of label l given X.
        """
        pass

    def predict_prob_l_given_y_X(self, X):
        """
        Predicts the probability of label l given input features X and y=1,
        i.e. p(l|y=1, X).

        Parameters:
            X (array-like): Input features for prediction.

        Returns:
            array-like: The predicted probability of label l given y=1 and X.
        """
        pass

    def predict_prob_y_given_X(self, X):
        """
        Predicts the probability of y given x.

        Parameters:
        X (array-like): The input data.

        Returns:
            array-like: The predicted probability of y given x.
        """
        pass
