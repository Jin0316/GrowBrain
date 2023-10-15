import torch
import torch.nn as nn
from config import ucf_config as config

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

class testmodel(nn.Module):
    def __init__(self): 
        super().__init__()
        self.linear1 = nn.Linear(32, 32)
        self.atten = nn.MultiheadAttention(32, 4)
        self.linear2 = nn.Linear(32, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x, _ = self.atten(x, x, x)
        x = self.relu(x)
        x = self.linear2(x)
        return x, _

class HyperFormer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, num_layers, num_channels, dropout=0.0, m_decom_dim = 5, emb_size = []):
        super().__init__()
        
        self.use_hyper = True

        self.n_blocks = 4
        self.groups = 2 
        self.m_dcom_heads = 2 
        self.channel_config = []
        self.classes, self.differences = [], []

        self.embed_dim = embed_dim
        self.out_size =  num_channels
        self.in_size =  num_channels
        self.m_decom_dim = m_decom_dim
        self.emb_size = emb_size

        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.classifiers = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)

        self.task_embedding, self.current_embedding = [], None 
        self.datasets = []

    def forward_feature(self, e):
        x = self.input_layer(e)
        self.temp_features = self.transformer(x)

    def forward(self, kernel_token, shape, index, block):
        h, k = self.emb_size[index]
        e = self.current_embedding.unsqueeze(dim = 0).to(config.device)
        ww = []
        for i in range(h):
            w = []
            for j in range(k):
                temp_kernel_token = kernel_token[i * k + j].to(config.device).unsqueeze(dim = 0)
                output = torch.cat([temp_kernel_token.unsqueeze(dim = 0), e], dim = 0)
                out_ = self.transformer(output)
                w1 = self.classifiers[str(block)+str(shape)+str(1)](out_[1])
                w2 = self.classifiers[str(block)+str(shape)+str(2)](out_[1])
                w3 = self.matrix_reconstruction(w1, w2)
                logit = self.matrix_match_size(w3, self.differences[block] * shape**2)
                logit = logit.reshape(self.channel_config[block][1], self.channel_config[block][0], int(shape), int(shape))
                w.append(logit)
            ww.append(torch.cat(w, dim = 1))
            
        hype_out =  torch.cat(ww, dim = 0)
        if self.use_hyper == True:
            return hype_out
        elif self.use_hyper == False:
            return hype_out.fill_(1)

    def init_classifiers(self, blocks, f_size):
        assert len(blocks) == len(f_size)

        self.channel_config = f_size
        classes, _ = self.init_cls_config(f_size)
        n_group = [3, 1]
        for index, block in enumerate(blocks):
            for group in n_group:
                # matrix decomposition 1 
                self.classifiers[str(block)+str(group)+str(1)] = nn.Sequential(
                                          nn.LayerNorm(self.embed_dim),
                                          nn.Linear(self.embed_dim, int(classes[index] * group * self.m_decom_dim), bias = True))

                # matrix decomposition 2 
                self.classifiers[str(block)+str(group)+str(2)] = nn.Sequential(
                                          nn.LayerNorm(self.embed_dim),
                                          nn.Linear(self.embed_dim, int(classes[index] * group * self.m_decom_dim), bias = True))

    def init_cls_config(self, f_size): 
        classes, difference = [], []
        assert type(f_size) == list 
        for f in f_size: 
            nearest_squre = self.find_nearest_sqre(f)
            if isinstance(nearest_squre, tuple):
                classes.append(nearest_squre[0])
                difference.append(nearest_squre[1])
            else: 
                classes.append(nearest_squre)
                difference.append(0)
        self.classes, self.differences = classes, difference
        print(self.classes, self.differences)
        return classes, difference

    def find_nearest_sqre(self, input):
        input_ = 1
        for i in input: 
            input_ *= i 
        a = torch.tensor(input_)
        if torch.sqrt(a) % 1 != 0:
            return int((torch.sqrt(a) // 1) + 1), int(((torch.sqrt(a) // 1) + 1)**2 - a)
        else: 
            return int(torch.sqrt(a))

    def matrix_reconstruction(self, w1, w2):
        w1 = w1.view(int(w1.shape[1] / self.m_decom_dim), self.m_decom_dim)
        w2 = w2.view(self.m_decom_dim, int(w2.shape[1] / self.m_decom_dim))
        return torch.matmul(w1, w2)

    def matrix_match_size(self, w, dif = torch.tensor(0)):
        f_w = torch.flatten(w)
        if dif == 0: 
            f_s_w = f_w
        elif dif != 0 :
            try:
                f_s_w = f_w[:-int(dif.item())]
            except:    
                f_s_w = f_w[:-int(dif)]
        return f_s_w

    def add_dataset(self, dataset):
        if dataset not in self.datasets:
            print('[vit_hyper.py]: add dataset {}'.format(dataset))
            self.datasets.append(dataset)
            self.task_embedding.append(nn.Parameter(torch.randn(1, self.embed_dim)))

    def set_dataset(self, dataset): 
        print('[vit_hyper.py]: set dataset {}'.format(dataset))
        self.current_embedding = self.task_embedding[self.datasets.index(dataset)]
