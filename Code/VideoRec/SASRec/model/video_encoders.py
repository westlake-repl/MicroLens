import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

class VideoMaeEncoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(VideoMaeEncoder, self).__init__()
        self.video_net = video_net
        # self.activate = nn.Tanh()
        self.activate = nn.GELU()
        self.args = args
        self.avg_pool = nn.AdaptiveAvgPool2d((1, args.word_embedding_dim))
        self.padding_label = torch.Tensor([-1]).to(args.local_rank)
        # self.add_nor = nn.LayerNorm(args.word_embedding_dim, eps=1e-6) ### 
        self.linear = nn.Linear(args.word_embedding_dim, args.embedding_dim)

    def forward(self, item_content):
        # torch.Size([112, 4, 3, 224, 224])
        item_scoring = self.video_net(item_content).last_hidden_state
        # torch.Size([112, 392, 768])
        item_scoring = self.avg_pool(item_scoring)
        # torch.Size([112, 1, 768])
        item_scoring = self.linear(item_scoring.squeeze(1)) # torch.Size([112, 512])
        # item_scoring = self.linear(self.add_nor(item_scoring.view(item_scoring.shape[0], -1))) 
        return self.activate(item_scoring)

class R3D18Encoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(R3D18Encoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        num_fc_ftr = self.video_net.fc.in_features
        self.video_net.fc = nn.Linear(num_fc_ftr, args.embedding_dim)
        xavier_normal_(self.video_net.fc.weight.data)
        if self.video_net.fc.bias is not None:
            constant_(self.video_net.fc.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(item_scoring)

class R3D50Encoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(R3D50Encoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(self.video_proj(item_scoring))

class C2D50Encoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(C2D50Encoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(self.video_proj(item_scoring))

class I3D50Encoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(I3D50Encoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(self.video_proj(item_scoring))

class CSN101Encoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(CSN101Encoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(self.video_proj(item_scoring))

class SLOW50Encoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(SLOW50Encoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(self.video_proj(item_scoring))

class EX3DSEncoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(EX3DSEncoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(self.video_proj(item_scoring))

class EX3DXSEncoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(EX3DXSEncoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(self.video_proj(item_scoring))

class X3DXSEncoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(X3DXSEncoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(self.video_proj(item_scoring))

class X3DSEncoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(X3DSEncoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(self.video_proj(item_scoring))

class X3DMEncoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(X3DMEncoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(self.video_proj(item_scoring))

class X3DLEncoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(X3DLEncoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(self.video_proj(item_scoring))

class MVIT16Encoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(MVIT16Encoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 400))
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        # print(item_content.shape)
        item_content = item_content.view(-1, 3, 224, 224)
        # print(item_content.shape)
        item_scoring = self.video_net(item_content)
        # print(item_scoring.shape)
        item_scoring = item_scoring.view(-1, self.args.frame_no, 400)
        # print(item_scoring.shape)
        item_scoring = self.avg_pool(item_scoring)
        # print(item_scoring.shape)
        # xxx
        return self.activate(self.video_proj(item_scoring.squeeze(1)))

class MVIT16X4Encoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(MVIT16X4Encoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(self.video_proj(item_scoring))

class MVIT32X3Encoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(MVIT32X3Encoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        item_scoring = self.video_net(item_content)
        return self.activate(self.video_proj(item_scoring))

class SLOWFAST50Encoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(SLOWFAST50Encoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        slow_item_content_1 = item_content[:, :, 0, :, :].unsqueeze(2)
        slow_item_content_2 = item_content[:, :, -1, :, :].unsqueeze(2)
        slow_item_content = torch.cat((slow_item_content_1, slow_item_content_2), 2)
        item_scoring = self.video_net([slow_item_content, item_content])
        return self.activate(self.video_proj(item_scoring))

class SLOWFAST16X8101Encoder(torch.nn.Module):
    def __init__(self, video_net, args):
        super(SLOWFAST16X8101Encoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.args = args
        self.video_proj = nn.Linear(400, args.embedding_dim)
        xavier_normal_(self.video_proj.weight.data)
        if self.video_proj.bias is not None:
            constant_(self.video_proj.bias.data, 0)

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        slow_item_content_1 = item_content[:, :, 0, :, :].unsqueeze(2)
        slow_item_content_2 = item_content[:, :, -1, :, :].unsqueeze(2)
        slow_item_content = torch.cat((slow_item_content_1, slow_item_content_2), 2)
        item_scoring = self.video_net([slow_item_content, item_content])
        return self.activate(self.video_proj(item_scoring))