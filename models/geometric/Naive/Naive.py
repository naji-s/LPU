import LPU.models.Kernel_MPE_grad_threshold
import LPU.models.geometric.Elkan.Elkan
EPSILON = 1e-16

class Naive(LPU.models.geometric.Elkan.Elkan):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """
    def __init__(self, config, *args, **kwargs):
        self.config = config
        super().__init__(config=self.config, **kwargs)

    def set_C(self, holdout_dataloader=None):        
        self.C = 1