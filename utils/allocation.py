import torch 
import copy 

class Allocator(object):
    def __init__(self, model, prune_ratio, mask, device):
        self.model = model
        self.prune_ratio = prune_ratio
        self.masks = mask
        self.device = device
        valid_key = list(self.masks.keys())[0] #return first key of masks
        self.current_dataset_idx = self.masks[valid_key].max()     
        self.generated_weights = None
    
    def pruning_mask_weights(self, weights, mask, layer_name):
        mask = mask.to(self.device)
        tensor = weights[mask.eq(self.current_dataset_idx)] 
        abs_tensor = tensor.abs()
        cutoff_rank = round(self.prune_ratio * tensor.numel())
        cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0]
        remove_mask = weights.abs().le(cutoff_value) * mask.eq(self.current_dataset_idx)

        # to avoid the malfunction
        if remove_mask.eq(1).sum() / tensor.numel() > self.prune_ratio + 0.1:
            random_mask = torch.randn(mask.shape).to(self.device)
            random_tensor = random_mask[mask.eq(self.current_dataset_idx)]
            cutoff_rank_ = round(self.prune_ratio * random_tensor.numel())
            cutoff_value_ = random_tensor.view(-1).cpu().kthvalue(cutoff_rank_)[0]
            
            random_remove_mask = random_mask.le(cutoff_value_) * mask.eq(self.current_dataset_idx)
            mask[random_remove_mask.eq(1)] = 0
        
        else:  
            mask[remove_mask.eq(1)] = 0

        print('Layer #%s, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_name, mask.eq(0).sum(), tensor.numel(),
               100 * mask.eq(0).sum() / tensor.numel(), weights.numel()))
        print('current weight values | max : {:.3f} | min : {:.3f} '.format(tensor.max(), tensor.min()))
        return mask

    def prune(self):
        print('Pruning for dataset idx: %d' % (self.current_dataset_idx))
        print('Pruning each layer by removing %.2f%% of values' % (100 * self.prune_ratio))
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            if 'conv' in n: 
                if n !='conv1': 
                    n = n.replace('resnet18__', '')
                    
                if p.requires_grad:
                    p_ = p.detach().clone()
                    mask = self.pruning_mask_weights(p_, self.masks[n], n)
                    self.masks[n] = mask.to(self.device)
                    p = p.detach()
                    p[self.masks[n].eq(0)] = 0.0
    
    def make_grads_zero(self):
        assert self.masks
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                if 'conv' in n: 
                    if n !='conv1': 
                        n = n.replace('resnet18__', '')
                    layer_mask = self.masks[n]
                    if p.grad is not None:
                        p.grad.data[layer_mask.ne(self.current_dataset_idx)] = 0
    
    def make_pruned_zero(self):
        assert self.masks
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            if 'conv' in n:
                if n !='conv1': 
                    n = n.replace('resnet18__', '')
                layer_mask = self.masks[n]
                p_clone = p.detach()
                p_clone[layer_mask.eq(0)] = 0.0

    def initialize_new_mask(self):
        """
            Turns previously pruned weights into trainable weights for
            current dataset.
        """
        self.current_dataset_idx += 1
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            if 'conv' in n:
                if n !='conv1': 
                    n = n.replace('resnet18__', '')
                mask = self.masks[n]
                mask[mask.eq(0)] = self.current_dataset_idx

    def apply_mask(self, task_index):
        print('[allocation.py] apply mask : task index : {}'.format(task_index))
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            if 'conv' in n:
                if n !='conv1': 
                    n = n.replace('resnet18__', '')
                mask = self.masks[n].to(self.device)
                p_clone = p.detach()
                p_clone[mask.eq(0)] = 0.0
                p_clone[mask.gt(int(task_index))] = 0.0 
    
    def initialize_new_masked_weights(self, dataset_idx):
        for n, p in self.model.resnet18.named_parameters():
            n = n.replace('.', '__')
            if 'conv' in n:
                weight = p.detach()
                mask = self.masks[n].to(self.device)
                random_init = 0.001 * torch.randn((p.size())).to(self.device)
                weight[mask.eq(dataset_idx)] = random_init[mask.eq(dataset_idx)]

    def apply_generated_mask(self, generated_weights):
        self.generated_weights = generated_weights
        assert self.generated_weights != None, 'Get generated weights'
        print('[allocation.py] apply generated parameters for task {}'.format(self.current_dataset_idx))
        for n, p in self.model.resnet18.named_parameters():
            n = n.replace('.', '__')
            if 'conv' in n:
                weight = p.detach()
                g_w = torch.sigmoid(self.generated_weights[n])
                g_w = g_w.masked_fill(self.masks[n].eq(self.current_dataset_idx), 1.0)
                g_w = g_w.masked_fill(self.masks[n].eq(0), 0.0)
                g_w = g_w.masked_fill(self.masks[n].gt(self.current_dataset_idx), 0.0)
                weight = weight * g_w

    def concat_original_model(self, dataset_idx, original_model):
        for (n, p), (original_n, original_p) in zip(self.model.named_parameters(), original_model.named_parameters()):
            n = n.replace('.', '__')
            if 'conv' in n:
                if n !='conv1': 
                    n = n.replace('resnet18__', '')
                weight = p.detach()
                original_weight = original_p.detach()
                mask = self.masks[n].to(self.device)
                mask_ = mask.lt(dataset_idx).__and__(mask.gt(0))
                weight[mask_.eq(1)] = original_weight[mask_.eq(1)]
