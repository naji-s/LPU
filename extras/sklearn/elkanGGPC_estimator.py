import LPU.models.geometric.geometric_base

class ElkanEstimator(LPU.models.geometric.geometric_base.GeometricGPLPUBase):
    def __init__(self, hold_out_ratio=0.1, **kwargs):
        raise NotImplementedError("ElkanEstimator is not fully implemented yet.")
    def fit(self, X=None, y=None, n_inducing_points = 100):
        """
        L: whether a data point is labeled or not
        """
        X, l, y = self._separate_labels(X, y)        

        # create a hold out set for estimating p(l=1|y=1)=c
        positives = np.where(l == 1.)[0]
        hold_out_size = int(np.ceil(len(positives) * self.hold_out_ratio))
        print('hold_out_size', hold_out_size)

        if len(positives) <= hold_out_size:
            raise('Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')
        
        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]

        #Hold out test kernel matrix
        X_test_hold_out = X[hold_out]
        keep = list(set(np.arange(len(l))) - set(hold_out))

        #New training data
        X = X[keep]
        l = l[keep]

        # if y is not None, we also only 
        if y is not None:
            y = y[keep]

        self._setup(X, l, y)

        shuffled_indices = torch.randperm(X.size(0))
        # Select the first n_inducing_points based on the shuffled indices
        inducing_points = X[shuffled_indices[:n_inducing_points]]    


        # self.gp_model = self._initialize_gp_model(inducing_points)
        # self.likelihood = ElkanLikelihood(X)
        # self.mll = gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self.gp_model, num_data=len(X), beta=1).to(self.device)
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.gp_model.parameters()},
        #     {'params': self.likelihood.parameters()},
        # ], lr=0.05)

        super().fit(X, l, y)

        self.C = self.predict_prob_l_given_x(X_test_hold_out).mean()

        # Load best gp_model state if available
        if best_model_state:
            self.gp_model.load_state_dict(best_model_state["gp"])
            self.likelihood.load_state_dict(best_model_state["likelihood"])

        self.gp_model.eval()
        self.likelihood.eval()

    def predict_proba(self, X):
        self.gp_model.eval()
        self.likelihood.eval()
        self.likelihood.update_input_data(X)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.gp_model(X).mean
            return self.likelihood(preds).mean

    def predict_y_proba_given_X(self, X):
        return self.predict_prob_l_given_x(X) / self.C
