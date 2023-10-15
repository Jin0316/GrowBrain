import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from config import ucf_config as config
from .resnet_v2 import no_downsample_BasicBlock, downsample_BasicBlock
from .operations import ConsensusModule
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from vit_hyper import HyperFormer
import copy

class Embedding(nn.Module):
    def __init__(self, z_num, z_dim):
        super(Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim
        h, k = self.z_num
        for i in range(h):
            for j in range(k):
                self.z_list.append(nn.parameter.Parameter(torch.randn(self.z_dim), requires_grad = True))
    def forward(self, list_index):
        return self.z_list[list_index]

def make_embedding(z_num, z_dim): 
    z_list = nn.ParameterList()
    z_num = z_num
    z_dim = z_dim

    h, k = z_num

    for i in range(h):
        for j in range(k):
            z_list.append(nn.parameter.Parameter(torch.randn(z_dim), requires_grad = True))

    return z_list

class CL_Learner(nn.Module):
    def __init__(self, num_segments = 3, modality = 'RGB', new_length = None, 
                       consensus_type = 'avg', before_softmax = True, crop_num = 1): 
        """
        Embedding 
        """
        super(CL_Learner, self).__init__()
        ########## Video action recognition ##########
        self.reshape = True
        self.num_segments = num_segments 
        self.modality = modality
        self.new_length = new_length
        self.consensus_type = consensus_type
        self.consensus = ConsensusModule(consensus_type)
        self.before_softmax = before_softmax
        self.crop_num = crop_num
        if not before_softmax and consensus_type != 'avg':
            raise ValueError('Only avg consensus can be used after Softmax')
        if new_length is None:
            self.new_length = 1 if modality =='RGB' else 5 
        else: 
            self.new_length == new_length

        print(("""
        Initializing TSN with base model: {} 
        Continual learning method: {} 
        modality : {} 
        num segments :{} 
        new_length: {} 
        consensus_module: {} 
        dropout ratio: {}""".format('resnet34', 'growbrain', self.modality, self.num_segments, self.new_length, consensus_type, 0)))

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        ########## Hypernetwork setting ##########        
        self.embed_dim = 64
        self.hidden_dim = 128
        self.num_channels = 32
        self.num_heads = 8
        self.num_layers = 7
        self.drop_out = 0.2 
        self.m_decom_dim = 7
        self.no_hyper = False

        ########## Continual learning setting ##########
        self.datasets, self.classifiers = [], nn.ModuleList()
        self.classifier = None 
        self.batchnorms, self.batchnorm = nn.ModuleList(), None
        self.initbns, self.initbn = nn.ModuleList(), None
        self.mask, self.layer_names, self.current_task_index = None, None, None 
        self.score = {}

        ########## Base, hypernetwork setting ##########
        self.conv1 = nn.parameter.Parameter(torch.randn(64, 3, 7, 7), requires_grad=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.block_conf = {'64' : [0,1,2], 
                           '128': [3,4,5,6],
                           '256': [7,8,9,10,11,12], 
                           '512': [13,14,15]}

        self.filter_size =  [[3, 64, 7],                                    # first conv in ResNet
                            [64, 64, 3],  [64, 64, 3],                      # ResNet layer 1                    
                            [64, 64, 3],  [64, 64, 3],
                            [64, 64, 3],  [64, 64, 3],

                            [64, 128, 3],  [128, 128, 3], [64, 128, 1],     # ResNet layer 2 
                            [128, 128, 3], [128, 128, 3],                  # 4
                            [128, 128, 3], [128, 128, 3],                  # 5
                            [128, 128, 3], [128, 128, 3],                  # 6

                            [128, 256, 3], [256, 256, 3], [128, 256 ,1],    # Resnet layer 3 
                            [256, 256, 3], [256, 256, 3],
                            [256, 256, 3], [256, 256, 3], 
                            [256, 256, 3], [256, 256, 3],
                            [256, 256, 3], [256, 256, 3],
                            [256, 256, 3], [256, 256, 3],

                            [256, 512, 3], [512, 512, 3], [256, 512, 1],    # ResNet layer 4
                            [512, 512, 3], [512, 512, 3],
                            [512, 512, 3], [512, 512, 3]]
        # the nuber of layer embeddings 
        self.z_size = [[1,1], [1,1],
                       [1,1], [1,1],
                       [1,1], [1,1],

                       [1, 1], [1, 2], [1, 1],
                       [1, 2], [1, 2],
                       [1, 2], [1, 2],
                       [1, 2], [1, 2],

                       [1, 1], [1, 2], [1, 1],
                       [1, 2], [1, 2],
                       [1, 2], [1, 2],
                       [1, 2], [1, 2],
                       [1, 2], [1, 2],
                       [1, 2], [1, 2],

                       [1, 1], [1, 2], [1, 1],
                       [1, 2], [1, 2],
                       [1, 2], [1, 2],]
        
        ########## Hypernetwork ##########
        self.hypernetwork = HyperFormer(self.embed_dim, self.hidden_dim, self.num_heads, 
                                        self.num_layers, self.num_channels, self.drop_out, 
                                        self.m_decom_dim, self.z_size) 
        # Make layer embedings
        self.embedding_list = []
        for i in range(len(self.z_size)):  
            self.embedding_list.append(make_embedding(self.z_size[i], self.embed_dim))

        # Make base Network; ResNet34 basenetwork in UCF-101 video action recognition scenario 
        self.resnet18 = nn.ModuleList()
        self.resnet18.append(no_downsample_BasicBlock(64, 64, stride = 1))
        self.resnet18.append(no_downsample_BasicBlock(64, 64))
        self.resnet18.append(no_downsample_BasicBlock(64, 64))

        self.resnet18.append(downsample_BasicBlock(64, 128, stride = 2))
        self.resnet18.append(no_downsample_BasicBlock(128, 128))
        self.resnet18.append(no_downsample_BasicBlock(128, 128))
        self.resnet18.append(no_downsample_BasicBlock(128, 128))

        self.resnet18.append(downsample_BasicBlock(128, 256, stride = 2))
        self.resnet18.append(no_downsample_BasicBlock(256, 256))
        self.resnet18.append(no_downsample_BasicBlock(256, 256))
        self.resnet18.append(no_downsample_BasicBlock(256, 256))
        self.resnet18.append(no_downsample_BasicBlock(256, 256))
        self.resnet18.append(no_downsample_BasicBlock(256, 256))

        self.resnet18.append(downsample_BasicBlock(256, 512, stride = 2))
        self.resnet18.append(no_downsample_BasicBlock(512, 512))
        self.resnet18.append(no_downsample_BasicBlock(512, 512))

    
    def forward(self, x):
        sample_len = (3 if self.modality == 'RGB' else 2) * self.new_length
        hyper_output = {}

        out = F.relu(self.bn1(F.conv2d(x.view((-1, sample_len) + x.size()[-2:]), self.conv1, stride = 2, padding=3)))
        out = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)(out)
        index = 0
        for n_block in range(len(self.resnet18)):  
            if n_block == 3 or n_block == 7 or n_block == 13: 
                if self.no_hyper == False:
                    self.resnet18[n_block].no_hyper_mode = False
                    g_conv1 = self.hypernetwork(self.embedding_list[index],   3, index,     self.make_group_config(n_block) )
                    g_conv2 = self.hypernetwork(self.embedding_list[index+1], 3, index + 1, self.make_group_config(n_block) )
                    g_conv3 = self.hypernetwork(self.embedding_list[index+2], 1, index + 2, self.make_group_config(n_block) )
                    g_conv1, g_conv2, g_conv3 = self.modify_g_w(self.layer_names[index], g_conv1),   \
                                                self.modify_g_w(self.layer_names[index+1], g_conv2), \
                                                self.modify_g_w(self.layer_names[index+2], g_conv3)
                    hyper_output[index]  =g_conv1
                    hyper_output[index+1]=g_conv2
                    hyper_output[index+2]=g_conv3
                    index += 3
                elif self.no_hyper == True:
                    self.resnet18[n_block].no_hyper_mode = True
                    g_conv1, g_conv2, g_conv3 = None, None, None

                out = self.resnet18[n_block](out, g_conv1, g_conv2, g_conv3)
            else: 
                if self.no_hyper == False:
                    self.resnet18[n_block].no_hyper_mode = False
                    g_conv1 = self.hypernetwork(self.embedding_list[index],   3, index,     self.make_group_config(n_block))
                    g_conv2 = self.hypernetwork(self.embedding_list[index+1], 3, index + 1, self.make_group_config(n_block))
                    g_conv1, g_conv2 = self.modify_g_w(self.layer_names[index], g_conv1),   \
                                       self.modify_g_w(self.layer_names[index+1], g_conv2)
                    hyper_output[index]   = g_conv1
                    hyper_output[index+1] = g_conv2
                    index += 2
                elif self.no_hyper == True: 
                    self.resnet18[n_block].no_hyper_mode = True
                    g_conv1, g_conv2 = None, None

                out = self.resnet18[n_block](out, g_conv1, g_conv2)

        out = nn.AdaptiveAvgPool2d((1,1))(out)
        out = out.view(out.size(0), -1)
        out = torch.nn.Dropout(0.25)(out)
        out = self.classifier(out)
        
        if not self.before_softmax:
            out = self.softmax(out)
        if self.reshape: 
            out = out.view((-1, self.num_segments) + out.size()[1:])
        out = self.consensus(out)

        return out.squeeze(1), hyper_output

    def forward_hypernetwork(self, x, index, shape):
        """
        forward pass of the hyper network: given embedding, shape and index 
        """
        output = self.hypernetwork(x, kernel_token = self.embedding_list[index], shape = shape, index = index)
        return output 
        
    def device_setting(self, device):
        self.hypernetwork.to(device)
        for classifier in self.hypernetwork.classifiers:
            print(classifier)
            classifier.to(device)
    
    def get_generated_weights(self):
        assert self.layer_names, 'layer names are required'
        assert self.mask, 'mask required'
        self.hypernetwork.eval()
        self.no_hyper = False
        self.hypernetwork.use_hyper = True
        generated_parameters = {}
        index = 0
        for n_block in range(len(self.resnet18)):  

            if n_block == 3 or n_block == 7 or n_block == 13: 
                generated_parameters[self.layer_names[index]]   =  self.hypernetwork(self.embedding_list[index],   3, index,     self.make_group_config(n_block))
                generated_parameters[self.layer_names[index+1]] =  self.hypernetwork(self.embedding_list[index+1], 3, index + 1, self.make_group_config(n_block))
                generated_parameters[self.layer_names[index+2]] =  self.hypernetwork(self.embedding_list[index+2], 1, index + 2, self.make_group_config(n_block))
                index += 3
            else: 
                generated_parameters[self.layer_names[index]]   =  self.hypernetwork(self.embedding_list[index],   3, index,     self.make_group_config(n_block))
                generated_parameters[self.layer_names[index+1]] =  self.hypernetwork(self.embedding_list[index+1], 3, index + 1, self.make_group_config(n_block))
                index += 2

        self.no_hyper = True
        self.hypernetwork.use_hyper = False
        return generated_parameters


    def modify_g_w(self, layer_name, g_w):
        # evolve preious set of parameters
        assert self.current_task_index > 1, "Use this funciton after train task 1. OR check task index is None"
        g_w = torch.sigmoid(g_w)
        g_w = g_w.masked_fill(self.mask[layer_name].eq(self.current_task_index), 1.0)
        g_w = g_w.masked_fill(self.mask[layer_name].eq(0), 0.0)
        g_w = g_w.masked_fill(self.mask[layer_name].gt(self.current_task_index), 0.0)
        return g_w

    def main_eval_mode(self, mode = False):
        for _, module in self.resnet18.named_modules():
            module.eval()

    def get_layer_names(self):
        if self.mask is not None: 
            self.layer_names = list(self.mask.keys())
            if 'conv1' in self.layer_names:
                self.layer_names = self.layer_names[1:]

    def regul_get_prev_layer_tokens(self, ckpt_dir): 
        prev_layer_tokens = {}
        generated_params  = {}
        assert self.current_task_index != None, 'Please update the current task index' 
        assert self.current_task_index >= 3, 'Not now' 
        print('[learner.py] [Hypernetwork] Get previous inputs and targets to apply regularize')
        print('[learner.py] extract layer tokens and generated parameters', config.save_names[1:self.current_task_index])
        for prev_task in config.save_names[1:self.current_task_index-1]:
            prev_task = prev_task
            ckpt = torch.load(ckpt_dir + prev_task, map_location=config.device)
            model = ckpt['model']
            prev_layer_tokens[prev_task] = model.embedding_list

        prev_task_index = 1
        for lt in list(prev_layer_tokens.keys()):
            print('[learner.py] previous layer tokens [0] ',prev_layer_tokens[lt][0])
            self.eval()
            self.hypernetwork.eval()
            self.no_hyper = False
            self.hypernetwork.use_hyper = True

            self.embedding_list = prev_layer_tokens[lt]
            self.hypernetwork.current_embedding = self.hypernetwork.task_embedding[prev_task_index]
            print('[learner.py] embedding of {} : {}'.format(lt ,self.hypernetwork.current_embedding))
            generated_params[lt] = self.get_generated_weights()
            prev_task_index += 1 

        del model
        del ckpt 
        return prev_layer_tokens, generated_params 
    

    def check_generated_weights_info(self):
        generated_param_info={}
        index=0
        for n_block in range(len(self.resnet18)):
            if n_block == 3 or n_block == 7 or n_block == 13: 
                g_conv1=self.hypernetwork(self.embedding_list[index],   3, index,     self.make_group_config(n_block))
                g_conv2=self.hypernetwork(self.embedding_list[index+1], 3, index + 1, self.make_group_config(n_block))
                g_conv3=self.hypernetwork(self.embedding_list[index+2], 1, index + 2, self.make_group_config(n_block))
                generated_param_info[index]   = [g_conv1.max().item(), g_conv1.min().item(), g_conv1.mean().item()]
                generated_param_info[index+1] = [g_conv2.max().item(), g_conv2.min().item(), g_conv2.mean().item()]
                generated_param_info[index+2] = [g_conv3.max().item(), g_conv3.min().item(), g_conv3.mean().item()]
                index+=3
            else: 
                g_conv1=self.hypernetwork(self.embedding_list[index],   3, index,     self.make_group_config(n_block))
                g_conv2=self.hypernetwork(self.embedding_list[index+1], 3, index + 1, self.make_group_config(n_block))
                generated_param_info[index]   = [g_conv1.max().item(), g_conv1.min().item(), g_conv1.mean().item()]
                generated_param_info[index+1] = [g_conv2.max().item(), g_conv2.min().item(), g_conv2.mean().item()]
                index+=2
        return generated_param_info
    
    def hypernetwork_init_classifiers(self, blocks = [0, 1, 2, 3], f_size = [[64,64],[64,128],[128,256],[256,512]]):
        self.hypernetwork.init_classifiers(blocks, f_size)

    def make_group_config(self, n_block):
        if n_block in [0, 1, 2]: 
            return 0
        if n_block in [3, 4, 5, 6]:
            return 1
        if n_block in [7, 8, 9, 10, 11, 12]:
            return 2
        if n_block in [13, 14, 15]: 
            return 3

    def ti_add_dataset(self, dataset, num_outputs):
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(512, num_outputs, bias = True)) 

        initial_bn = nn.ModuleDict()
        for name, module in enumerate(self.resnet18.modules()):   # name : index 2 ,6 ... 
            if isinstance(module, nn.BatchNorm2d):
                new_bn = copy.deepcopy(module)
                # if extract imagenet 
                new_bn.weight.data.fill_(1)
                new_bn.bias.data.zero_()
                initial_bn[str(name)] = new_bn
        self.batchnorms.append(initial_bn)

        new_bn = copy.deepcopy(self.bn1)
        new_bn.weight.data.fill_(1)
        new_bn.bias.data.zero_()
        self.initbns.append(new_bn)

        self.hypernetwork.add_dataset(dataset)

    def save_bn(self):
        for name, module in enumerate(self.resnet18.modules()): # save bn value
            if isinstance(module, nn.BatchNorm2d):
                #print(name)
                self.batchnorm[str(name)].weight.data.copy_(module.weight.data)
                self.batchnorm[str(name)].bias.data.copy_(module.bias.data)
                self.batchnorm[str(name)].running_var.data.copy_(module.running_var.data)
                self.batchnorm[str(name)].running_mean.data.copy_(module.running_mean.data)

        self.initbn.weight.data.copy_(self.bn1.weight.data)
        self.initbn.bias.data.copy_(self.bn1.bias.data)
        self.initbn.running_var.data.copy_(self.bn1.running_var.data)
        self.initbn.running_mean.data.copy_(self.bn1.running_mean.data)

    def ti_set_dataset(self, dataset): 
        self.classifier = self.classifiers[self.datasets.index(dataset)]
        self.batchnorm = self.batchnorms[self.datasets.index(dataset)]
        self.initbn = self.initbns[self.datasets.index(dataset)]
        self.hypernetwork.set_dataset(dataset)

        self.bn1.weight.data.copy_(self.initbn.weight.data)
        self.bn1.bias.data.copy_(self.initbn.bias.data)
        self.bn1.running_var = self.initbn.running_var
        self.bn1.running_mean = self.initbn.running_mean

        for name, module in enumerate(self.resnet18.modules()): # change bn
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.copy_(self.batchnorm[str(name)].weight.data)
                module.bias.data.copy_(self.batchnorm[str(name)].bias.data)
                module.running_var = self.batchnorm[str(name)].running_var
                module.running_mean = self.batchnorm[str(name)].running_mean

    def ti_bn_eval(self, mode = True):
        super(CL_Learner, self).train(mode)

        for _, module in self.resnet18.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if mode == True: 
                    module.eval()
                if mode == False: 
                    module.train()
                    
    def get_score(self):
        gradient, self.score = {}, {}
        element = {}
        index=0
        with torch.no_grad():
            for block_index in range(len(self.resnet18)):
                if block_index == 3 or block_index == 7 or block_index == 13:
                    gradient[index]  =self.resnet18[block_index].conv1.detach() * self.resnet18[block_index].conv1.grad.detach() / self.resnet18[block_index].g_conv1
                    gradient[index+1]=self.resnet18[block_index].conv2.detach() * self.resnet18[block_index].conv2.grad.detach() / self.resnet18[block_index].g_conv2
                    gradient[index+2]=self.resnet18[block_index].conv3.detach() * self.resnet18[block_index].conv3.grad.detach() / self.resnet18[block_index].g_conv3
                    element[index]   = self.resnet18[block_index].g_conv1
                    element[index+1] = self.resnet18[block_index].g_conv2
                    element[index+2] = self.resnet18[block_index].g_conv3
                    index += 3
                else: 
                    gradient[index]  =self.resnet18[block_index].conv1.detach() * self.resnet18[block_index].conv1.grad.detach() / self.resnet18[block_index].g_conv1
                    gradient[index+1]=self.resnet18[block_index].conv2.detach() * self.resnet18[block_index].conv2.grad.detach() / self.resnet18[block_index].g_conv2
                    element[index]   = self.resnet18[block_index].g_conv1
                    element[index+1] = self.resnet18[block_index].g_conv2
                    index +=2
            index=0
            
            for index in list(gradient.keys()):
                before_norm = - (gradient[index] * element[index]) + (0.5 * element[index] **2 * gradient[index]**2) 
                after_norm = (before_norm - before_norm.min())
                self.score[index] = after_norm
            gradient, element = {}, {}
        return self.score
    
    def clear_score(self): 
        self.score = {}