import torchvision.models as models
import clip
from transformers import CLIPVisionModel
from REC.model.layers import ItemEncoder, FIXItemEncoder
from transformers import CLIPVisionModel,SwinModel,ViTMAEModel,SwinConfig,BeitModel
import torch
from REC.model.layers import *
class SLOWFAST50Encoder(torch.nn.Module):
    def __init__(self, video_net,embedding_dim):
        super(SLOWFAST50Encoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()
        self.video_proj = nn.Linear(400, embedding_dim)
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
    
def load_model(config):

    encoder_name = config['encoder_name']
    encoder_source = config['encoder_source']
    output_dim = config['embedding_size']
    tune_scale = config['fine_tune_arg']['tune_scale']
    con_pretrained = config['fine_tune_arg']['pre_trained']
    activation = config['fine_tune_arg']['activation']
    dnn_layers = config['fine_tune_arg']['dnn_layers']
    method = config['fine_tune_arg']['method'] if config['fine_tune_arg']['method'] else 'cls'

    if encoder_source == 'torchvision' or encoder_source == None:
        if encoder_name == 'resnet18':
            model = models.resnet18(pretrained=con_pretrained)
            input_dim = model.fc.in_features
    
        elif encoder_name == 'resnet34':
            model = models.resnet34(pretrained=con_pretrained)
            input_dim = model.fc.in_features
 
        elif encoder_name == 'resnet50':
            model = models.resnet50(pretrained=con_pretrained)
            input_dim = model.fc.in_features
        
        for index, (name, param) in enumerate(model.named_parameters()):
            if index < tune_scale:
                param.requires_grad = False
    
        model = ItemEncoder(item_encoder=model, input_dim=input_dim, output_dim=output_dim, act_name=activation,dnn_layers=dnn_layers)
    
    
    elif encoder_source == 'clip':
        if encoder_name == 'RN50':
            model, _ = clip.load(device='cpu', name=encoder_name)
            model = model.visual
            input_dim = model.attnpool.c_proj.out_features    #用什么输出比较好啊。。
            if tune_scale != 0:
                for index, (name, param) in enumerate(model.named_parameters()):
                    if index < 78 or index > 164:
                        param.requires_grad = False
        elif encoder_name == 'RN50x4':
            model, _ = clip.load(device='cpu', name=encoder_name)
            model = model.visual
            input_dim = model.attnpool.c_proj.out_features    #用什么输出比较好啊。。
            for index, (name, param) in enumerate(model.named_parameters()):
                #if index < 105 or index > 254: #two layer
                if index < 198 or index > 254:
                    param.requires_grad = False
        
        elif encoder_name == 'RN50x16':
            model, _ = clip.load(device='cpu', name=encoder_name)
            model = model.visual
            input_dim = model.attnpool.c_proj.out_features    #用什么输出比较好啊。。
            for index, (name, param) in enumerate(model.named_parameters()):
                if index < 306 or index > 380:
                    param.requires_grad = False
        
        elif encoder_name == 'RN50x64':
            model, _ = clip.load(device='cpu', name=encoder_name)
            model = model.visual
            input_dim = model.attnpool.c_proj.out_features    #用什么输出比较好啊。。
            for index, (name, param) in enumerate(model.named_parameters()):
                if index < 504 or index > 596:
                    param.requires_grad = False
        
        elif encoder_name == 'ViT-B/32':
            model, _ = clip.load(device='cpu',name=encoder_name)
            model = model.visual
            input_dim = 768    #用什么输出比较好啊。。
    
            for index, (name, param) in enumerate(model.named_parameters()):
                if index < tune_scale:
                    param.requires_grad = False
    
        #model = ItemEncoder(item_encoder=model, input_dim=input_dim, output_dim=output_dim, act_name=activation,dnn_layers=dnn_layers)
    
        if method == 'cls':
            model = ItemEncoder(item_encoder=model, input_dim=input_dim, output_dim=output_dim, act_name=activation,dnn_layers=dnn_layers)
        elif method == 'pool':
            input_dim = model.attnpool.c_proj.in_features
            model.attnpool = torch.nn.AdaptiveAvgPool2d(1)
            model = CLIPItemEncoder(item_encoder=model, input_dim=input_dim, output_dim=output_dim, act_name=activation,dnn_layers=dnn_layers)
    
    elif encoder_source == 'video':
        video_model = torch.hub.load('./pytorchvideo_rs', model='slowfast_r50', source='local',
                                         head_pool_kernel_sizes=((1, 7, 7), (4, 7, 7)), pretrained=True)

        for index, (name, param) in enumerate(video_model.named_parameters()):
            # print(index, name)
            if index < tune_scale:
                param.requires_grad = False
        model = SLOWFAST50Encoder(video_model, output_dim)
    
    
    
    
    elif encoder_source == 'transformers':       
        if encoder_name == 'clip-vit-base-patch32':
            
            if con_pretrained:
                model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32') 
                input_dim = 768
                
                for index, (name, param) in enumerate(model.named_parameters()):
                    if index < tune_scale:
                        param.requires_grad = False
            else:
                model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32') 
                input_dim = 768
                
                for index, (name, param) in enumerate(model.named_parameters()):
                    if index < tune_scale:
                        param.requires_grad = False
                    else:
                        param.data.normal_(mean=0.0, std=0.02)
                
            
            if method == 'cls':
                model.vision_model.post_layernorm = Identity()
                model = ClsItemEncoder(item_encoder=model, input_dim=input_dim, output_dim=output_dim, act_name=activation,dnn_layers=dnn_layers)
            
            elif method == 'mean':
                model.vision_model.post_layernorm = Identity()
                model = MeanItemEncoder(item_encoder=model, input_dim=input_dim, output_dim=output_dim, act_name=activation,dnn_layers=dnn_layers)

            elif method == 'pool':
                model = PoolItemEncoder(item_encoder=model, input_dim=input_dim, output_dim=output_dim, act_name=activation,dnn_layers=dnn_layers)


        elif encoder_name == 'vit-mae-base':
            input_dim = 768
            model = ViTMAEModel.from_pretrained('facebook/vit-mae-base')
            method = 'cls'
            for index, (name, param) in enumerate(model.named_parameters()):
                if index < tune_scale:
                    param.requires_grad = False
            model = ClsItemEncoder(item_encoder=model, input_dim=input_dim, output_dim=output_dim, act_name=activation,dnn_layers=dnn_layers)
        elif encoder_name == 'beit-base-patch16':
            input_dim = 768
            model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224')
    
            for index, (name, param) in enumerate(model.named_parameters()):
                if index < tune_scale:   #tune最后两层183
                    param.requires_grad = False
            model = PoolItemEncoder(item_encoder=model, input_dim=input_dim, output_dim=output_dim, act_name=activation,dnn_layers=dnn_layers)
        
        elif encoder_name == 'swin-tiny-patch4-window7-224':
            input_dim = 768
            if con_pretrained:
                model = SwinModel.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
            else:
                configuration = SwinConfig()
                model = SwinModel(configuration)
            for index, (name, param) in enumerate(model.named_parameters()):
                if index < tune_scale:
                    param.requires_grad = False
            if method == 'cls':
                model.pooler = Identity()
                model = ClsItemEncoder(item_encoder=model, input_dim=input_dim, output_dim=output_dim, act_name=activation,dnn_layers=dnn_layers)
            elif method == 'pool':   #SwinForImageClassification用的是这个
                model = PoolItemEncoder(item_encoder=model, input_dim=input_dim, output_dim=output_dim, act_name=activation,dnn_layers=dnn_layers)
    return model
    

def load_weights(config):
    image_feature_path = config['v_feat_path']
    device = config['device']
    output_dim = config['embedding_size']
    activation = config['fine_tune_arg']['activation']
    dnn_layers = config['dnn_layers']
  
    model = FIXItemEncoder(weight_path=image_feature_path, device=device
    , output_dim=output_dim, act_name=activation,dnn_layers=dnn_layers)

    return model