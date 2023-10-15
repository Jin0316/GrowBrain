import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.init as init 
import torch.utils.data as data
from torch.autograd import Variable
from utils.allocation import Allocator
import time 
from config import ucf_config as config
from torch.distributions import Categorical
import copy 

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return correct, target.size(0)


class Manager(object):
    def __init__(self, model, masks, train_loader, test_loader, task_idx, **kwargs): 
        self.model = model
        self.task_idx = task_idx

        self.train_loader = train_loader 
        self.test_loader  = test_loader  

        self.get_param = True
        self.retrain = False
        self.epochs    = kwargs['epochs']
        self.optimizer = kwargs['optimizer']
        self.hyper_optimizer = kwargs['hyper_optimizer']
        self.prune_ratio = kwargs['pruning_ratio']
        self.mask = masks 
        self.lr_sche = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[25, 40], gamma = 0.1)
        self.hyper_lr_sche = optim.lr_scheduler.MultiStepLR(self.hyper_optimizer, milestones=[5, 7], gamma = 0.2)
        
        self.criterion = kwargs['criterion']
        self.device    = kwargs['device']
        self.allocator = Allocator(self.model, self.prune_ratio, self.mask, device = self.device)
        self.generated_parameters = None
        self.original_model = None
        self.regul_generated_weights, self.regul_prev_layer_tokens, self.task_embeds = None, None, None

    def get_pretrained(self):
        pre_model = torch.load('resnet34_pretrained.pt', map_location=self.device)
        for (n, model_p), (n_, pre_p) in zip(self.model.resnet18.named_parameters(), pre_model.resnet18.named_parameters()):
            n = n.replace('.', '__')
            if 'conv' in n: 
                if n !='conv1': 
                    n = n.replace('resnet18__', '')
                model_p = model_p.detach()
                pre_p = pre_p.detach()
                mask = self.mask[n].to(self.device)
                mask_ = mask.eq(self.task_idx)
                print(model_p.shape, mask_.shape)
                model_p[mask_.eq(1)] = pre_p[mask_.eq(1)]

    def do_train(self, get_param = True, retrain = False):
        print('[Manger > do_train] | Current task index : {}'.format(self.task_idx))
        if self.task_idx > 1 and self.retrain == False: 
            self.model.mask = self.mask
            self.model.get_layer_names()
            self.model.current_task_index = self.task_idx
    
        self.model = self.model.to(self.device)
        self.model.train()

        if get_param == True: 
            self.get_param = True
        elif get_param == False: 
            self.get_param = False

        if retrain: 
            self.retrain = True
            self.epochs = 20
            self.optimzier = optim.SGD(self.model.parameters(),lr=1e-3,momentum=0.9)
            self.lr_sche = optim.lr_scheduler.MultiStepLR(self.optimizer, [5, 10, 15], gamma = 0.2)

        else: self.retrain = False
        
        if self.task_idx == 1: 
            self.model.hypernetwork.eval()
            self.model.hypernetwork.use_hyper = False 
            self.model.no_hyper = True

        elif self.task_idx != 1: 
            self.model.no_hyper = False
            self.model.hypernetwork.train()
            self.model.hypernetwork.use_hyper = True

        log = self.do_epoch()

        self.model.clear_score()
        return log 
        
    def do_epoch(self):
        epoch_generated_param_info = {}
        
        for epoch in range(self.epochs):
            hyper = False
            if self.retrain == False:
                self.original_model = copy.deepcopy(self.model)

            if self.retrain == True:
                self.model.train()
                self.model.hypernetwork.eval()      
                self.model.no_hyper = True
                self.model.hypernetwork.use_hyper = False
                if self.task_idx > 1: 
                    self.allocator.apply_generated_mask(self.generated_parameters)
                self.allocator.apply_mask(self.task_idx)
                hyper = False

            if (self.retrain == False and self.task_idx > 1):
                    
                if self.get_param:
                    self.model.hypernetwork.train()
                    self.model.no_hyper = False
                    self.model.hypernetwork.use_hyper = True
                    self.model.get_layer_names()
                    hyper = True
                
                elif self.get_param == False: #  epoch > 10: 
                    self.model.no_hyper = False
                    self.model.hypernetwork.use_hyper = True
                    self.model.hypernetwork.eval()
                    self.model.no_hyper = True
                    self.model.hypernetwork.use_hyper = False
                    self.allocator.apply_generated_mask(self.generated_parameters)
                    hyper = False

            correct = 0 
            total = 0 
            running_loss = 0
            elapse_time = 0 
            sparsity_loss = 0
            for step, (image, label) in enumerate(self.train_loader):
                loss, s_loss, cor, tot, t_time = self.do_batch(image, label, hyper, epoch)
                correct += cor 
                total += tot
                running_loss += loss
                sparsity_loss += s_loss
                elapse_time += t_time

            if self.task_idx > 1: 
                self.allocator.concat_original_model(self.task_idx, self.original_model)    
            if (self.retrain == False and self.task_idx > 1): 
                if epoch >=0 and epoch <=10: 
                    epoch_generated_param_info[epoch] = self.model.check_generated_weights_info()

            self.lr_sche.step()
            if hyper:
                self.hyper_lr_sche.step()
                
            print('Task idx {} | epoch {} | loss: {:.5f} | sparsity loss {:.3f} | acc: {:.3f} | time : {:.3f} ' \
                .format(self.task_idx, epoch + 1, running_loss/step, sparsity_loss/step, 100 * correct/total, elapse_time))
        
        try: 
            return epoch_generated_param_info[list(epoch_generated_param_info.keys())[-1]]
        except:
            return {}
    def do_batch(self, image, label, hyper, epoch):
        correct = 0.0
        total = 0
        running_loss = 0.0
        temp_time = 0 
        s_loss = 0
        image = image.to(self.device)
        label = label.to(self.device)
        self.model.zero_grad()
        image, label = Variable(image), Variable(label)
        start = time.time()
        pred, hyper_output = self.model(image)
        loss = self.criterion(pred, label)

        if hyper: 
            if epoch > 0 : 
                score = self.model.get_score()
            else:
                score = None
            s_loss = self.sparsity_loss(hyper_output, lambd = config.lambda_sp, score = score, epoch = epoch)
            loss += s_loss
        
        loss.backward()
        
        if self.task_idx > 1:
            self.allocator.make_grads_zero()
        self.optimizer.step()    
        if hyper: 
            self.hyper_optimizer.step()
        running_loss += loss.item()
        cor, tot = accuracy(pred.data, label)
        correct += cor.sum()
        total += tot
        end = time.time()
        temp_time += end-start
        return running_loss, s_loss, correct, total, temp_time

    def sparsity_loss(self, g_output, lambd=1.0, score = None, epoch = 0):
        s_loss = 0
        epsilon = 1e-5
        mask_keys = list(self.mask.keys())
        mask_index = 1
        score_index = 0

        binary_cross_entropy = nn.BCELoss()
        if score is not None:
            score_keys = list(score.keys())

        for index in list(g_output.keys()):
            mask = self.mask[mask_keys[mask_index]].lt(self.task_idx) * self.mask[mask_keys[mask_index]].gt(0)
            zero_mask = torch.zeros(mask.shape).to(mask.device)
            masked_g = g_output[index] * mask
            temp = binary_cross_entropy(masked_g, zero_mask)
            s_loss += lambd * temp.mean()
            
            if epoch > 0 :
                if score is not None and len(score) > 0: 
                    masked_score = score[score_keys[score_index]] * mask
                    masked_score = (masked_score - masked_score.min())/(masked_score.max() + 1e-5)
                    s_loss += lambd * torch.norm(masked_score - (masked_score * (masked_g / (masked_g + epsilon))), p = 1)
                    score_index += 1
                
            mask_index += 1 
        return s_loss

    def prune(self, save_name):
        self.save_model(save_name + '_before_prune')
        self.allocator.prune()
        self.do_train(retrain = True)
        self.save_model(save_name)
        
    def init_weight(self, network, task_idx):
        assert network in ['resnet18', 'hypernetwork', 'both']
        if network == 'resnet18':
            for n, p in self.model.resnet18.named_parameters():
                if 'conv' in n or 'shortcut.0' in n:
                    if task_idx == 1 :
                        init.xavier_normal_(p)
                    else: 
                        weight = p.detach()
                        weight = 0.001 * torch.randn(p.shape).to(self.device)
                        
                if 'bn' and 'weight' in n:
                    p.data.fill_(1)
                if 'bn' and 'bias' in n:
                    p.data.fill_(0)

        elif network == 'hypernetwork':
            for n, p in self.model.hypernetwork.named_parameters():
                if 'linear.weight' in n:
                    init.xavier_normal_(p)

        elif network == 'both':
            for n, p in self.model.named_parameters():
                if 'conv' in n:
                    if task_idx == 1 :
                        init.xavier_normal_(p)
                    else: 
                        weight = p.detach()
                        weight = 0.001 * torch.randn(p.shape).to(self.device)

                if '.weight' in n and 'hypernetwork' in n and 'linear' in n:
                    init.xavier_normal_(p)
                if 'bn' and 'weight' in n:
                    p.data.fill_(1)
                if 'bn' and 'bias' in n:
                    p.data.fill_(0)

    def save_model(self, savename):
        model = self.model  
        model.save_bn()
        if self.task_idx > 1: 
            ckpt = {'model' : model,
                    'masks': self.allocator.masks,
                    'generated_params': self.generated_parameters}
        else: 
            ckpt = {'model' : model,
                    'masks': self.allocator.masks}
        print('Saving model {}'.format(savename + '.pt'))
        torch.save(ckpt, str(savename) + '.pt')

    def do_eval(self, task_index): 
        self.model = self.model.to(self.device)
        self.model.no_hyper = True
        self.model.hypernetwork.use_hyper = False
        self.model.hypernetwork.eval()
        self.model.eval()
        if self.task_idx > 1: 
            self.allocator.apply_generated_mask(self.generated_parameters)
        self.allocator.apply_mask(task_index)

        total = 0
        correct = 0
        entropy = torch.tensor([]).to(config.device)
        print('Performing eval...')

        with torch.no_grad():
            for step, (image, label) in enumerate(self.test_loader):
                image = image.to(self.device)
                label = label.to(self.device)
                image, label = Variable(image), Variable(label)
                pred, _ = self.model(image)
                entropy = torch.cat([entropy, Categorical(torch.softmax(pred, dim = 1)).entropy()])
                
                cor, tot = accuracy(pred.data, label)
                correct += cor.sum()
                total += tot

        acc = 100 * correct / total
        print('acc: {:.3f}'.format(acc))
        return acc, entropy

    def do_epoch_regul(self, regul_epochs, prev_layer_tokens, generated_weights):
        print('[train.py] > regularization on hypernetwork')
        self.regul_prev_layer_tokens, self.regul_generated_weights = prev_layer_tokens, generated_weights 
        optimizer = optim.SGD(self.model.hypernetwork.parameters(), lr = 0.1, momentum= 0.9, weight_decay=0)
        lr_sche   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100)

        assert self.regul_generated_weights != None 
        self.regul_prev_layer_tokens[config.dataset_list[self.task_idx-1] + '.pt'] = self.model.embedding_list
        self.task_embeds = self.model.hypernetwork.task_embedding[1:]
        self.regul_generated_weights[config.dataset_list[self.task_idx-1] + '.pt'] = self.generated_parameters

        print('[train.py] > task embeddings ',self.task_embeds)
        
        for g_w in list(self.regul_generated_weights.keys()):
            prev_g_w = self.regul_generated_weights[g_w]
            print('[train.py] {} generated weights {}'.format(g_w, prev_g_w['0__conv1'][0][0]))
        
        for epoch in range(regul_epochs):
            r_loss = self.regul_hyper()
            self.model.hypernetwork.zero_grad()
            r_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.hypernetwork.parameters(), 2)
            optimizer.step()
            print('regularization step | epoch {} | loss {}'.format(epoch, r_loss.item()))
            lr_sche.step()

    def regul_hyper(self): 
        # Make Names
        self.model.eval()
        self.model.hypernetwork.train()

        prev_task_name = config.save_names[1:self.task_idx]
        for lt in list(self.regul_prev_layer_tokens.keys()):
            if lt == prev_task_name[-1]:
                for lt_ in self.regul_prev_layer_tokens[lt]: 
                    self.regul_embd_train(lt_, True)
            else:
                for lt_ in self.regul_prev_layer_tokens[lt]: 
                    self.regul_embd_train(lt_, False)

        # Get task embeddings 
        for te in self.task_embeds:
            self.regul_embd_train(te, False)
    
        self.regul_embd_train(self.task_embeds[-1], True)

        assert self.task_idx >= 3, 'The regularization on hypernetwork should apply task idx >= 3'        
        assert list(self.regul_generated_weights.keys()) == prev_task_name

        loss = 0
        
        task_number = 0
        for prev_task in prev_task_name: 
            if prev_task == prev_task_name[-1]: 
                beta = 1.0 
            else: 
                beta = 0.1 

            self.model.hypernetwork.use_hyper = True
            generated_parameters = {}
            self.model.hypernetwork.current_embedding = self.task_embeds[task_number]
            index = 0
            for n_block in range(len(self.model.resnet18)):  
                if n_block == 3 or n_block == 7 or n_block == 13: 
                    generated_parameters[index]     = self.model.hypernetwork(self.regul_prev_layer_tokens[prev_task][index],   3, index, self.make_group_config(n_block))
                    generated_parameters[index + 1] = self.model.hypernetwork(self.regul_prev_layer_tokens[prev_task][index+1], 3, index + 1 , self.make_group_config(n_block))
                    generated_parameters[index + 2] = self.model.hypernetwork(self.regul_prev_layer_tokens[prev_task][index+2], 1, index + 2 , self.make_group_config(n_block))
                    index += 3
                else: 
                    generated_parameters[index]     = self.model.hypernetwork(self.regul_prev_layer_tokens[prev_task][index],   3, index, self.make_group_config(n_block))
                    generated_parameters[index + 1] = self.model.hypernetwork(self.regul_prev_layer_tokens[prev_task][index+1], 3, index + 1, self.make_group_config(n_block))
                    index += 2

            targets = self.regul_generated_weights[prev_task]
            generated_param_keys = list(generated_parameters.keys())
            target_key_list = list(targets.keys())
            mask_key_list = list(self.mask.keys())
            if len(target_key_list) != len(mask_key_list):
                mask_key_list = mask_key_list[1:]
            m_idx = 0

            for g_p_key, target_key in zip(generated_param_keys, target_key_list):
                loss += beta * self.regul_hyper_loss(generated_parameters[g_p_key], targets[target_key], self.mask[mask_key_list[m_idx]], task_number + 2)
                m_idx += 1 
            task_number += 1
        return loss 

    def regul_embd_train(self, embeds, mode = False):
        try:
            for embed in range(embeds):
                embed.requires_grad = mode
        except: 
            embeds.requires_grad = mode 
    
    def regul_hyper_loss(self, w, w_, mask, task_index):
        w_ = w_.detach().clone()
        
        loss = nn.MSELoss()
        w[mask.eq(0)] = 0 
        w[mask.ge(task_index)] = 0

        w_[mask.eq(0)] = 0 
        w_[mask.ge(task_index)] = 0
        return loss(w, w_)

    
    def make_group_config(self, n_block):
        if n_block in [0, 1, 2]: 
            return 0
        if n_block in [3,4,5,6]:
            return 1
        if n_block in [7,8,9,10,11,12]:
            return 2
        if n_block in [13,14,15]: 
            return 3