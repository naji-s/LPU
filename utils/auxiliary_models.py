import torch.nn

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(MultiLayerPerceptron, self).__init__()

        self.input_dim = input_dim
        self.l1 = torch.nn.Linear(self.input_dim, 300, bias=False)
        self.b1 = torch.nn.BatchNorm1d(300)
        self.l2 = torch.nn.Linear(300, 300, bias=False)
        self.b2 = torch.nn.BatchNorm1d(300)
        self.l3 = torch.nn.Linear(300, 300, bias=False)
        self.b3 = torch.nn.BatchNorm1d(300)
        self.l4 = torch.nn.Linear(300, 300, bias=False)
        self.b4 = torch.nn.BatchNorm1d(300)
        self.l5 = torch.nn.Linear(300, output_dim)
        self.af = torch.nn.functional.relu
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        h = self.l1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.l2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.l3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.l4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.l5(h)
        return h