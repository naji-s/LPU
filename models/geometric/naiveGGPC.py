import lpu.models.Kernel_MPE_grad_threshold
import lpu.models.geometric.elkanGGPC
EPSILON = 1e-16

class NaiveGGPC(lpu.models.geometric.elkanGGPC.ElkanGGPC):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """
    def __init__(self, config, *args, **kwargs):
        self.config = config
        super().__init__(config=self.config, **kwargs)

    def set_C(self, holdout_dataloader=None):        
        self.C = 1