import torch
import torch.nn as nn

class SumFusion(nn.Module):
    def __init__(self, args):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_y = nn.Linear(args.embedding_dim, args.embedding_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return output

class ConcatFusion(nn.Module):
    def __init__(self, args):
        super(ConcatFusion, self).__init__()
        self.fc_1 = nn.Linear(args.embedding_dim * 3, args.embedding_dim)
        self.fc_2 = nn.Linear(args.embedding_dim, args.embedding_dim)

    def forward(self, x, y, z):
        output = torch.cat((x, y, z), dim=1)
        output = self.fc_2(self.fc_1(output))
        return output

class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, args, x_film=True):
        super(FiLM, self).__init__()

        self.dim = args.embedding_dim
        self.fc = nn.Linear(args.embedding_dim, 2 * args.embedding_dim)
        self.fc_out = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.x_film = x_film

    def forward(self, x, y):
        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, args, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_y = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.fc_out = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return output

