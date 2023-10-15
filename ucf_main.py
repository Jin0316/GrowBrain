import torch
import torch.nn as nn 
import torch.optim as optim

import os 
from utils.train import Manager
from models.learner import CL_Learner
from config import ucf_config as config
from dataset.dataloader import TSNDataSet

device = config.device
current_path = os.getcwd()
dataset_list = config.dataset_list
classes = config.dataset_classes

task_incremental_acc = {}

train_ = TSNDataSet(config.root_path, config.pickle_file,
                    task_index=0, num_segments=config.num_segments, new_length=1, 
                    modality='RGB', image_tmpl='img_{:05d}.jpg', 
                    transform=config.transform, force_grayscale=False, 
                    random_shift=True, mode='train')

test_ = TSNDataSet(config.root_path, config.pickle_file, 
                    task_index=0, num_segments=config.num_segments, new_length=1, 
                    modality='RGB', image_tmpl='img_{:05d}.jpg', 
                    transform=config.transform, force_grayscale=False, 
                    random_shift=True, mode='test')


for task_idx in range(0, len(dataset_list)):
    current_task = dataset_list[task_idx]
    task_idx = task_idx + 1

    n_cls = classes[task_idx -1]
    print('Train for task index {} | task name {} | classes {} |'.format(task_idx, current_task, n_cls))

    if task_idx == 1:
        pre_ = torch.load(config.ckeck_point_dir + 'ckpt_ucf0.pt', map_location = device)
        model = CL_Learner()
        model.load_state_dict(pre_.state_dict(), strict=False)
        masks = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                if 'conv' in n: 
                    if n !='conv1': 
                        n = n.replace('resnet18__', '')
                    masks[n] = torch.randn(p.shape).fill_(1)

    else: 
        print('Load previous model | load name: {}'. format(dataset_list[task_idx - 2]))
        load_name = str(dataset_list[task_idx-2]) + '.pt'
        ckpt = torch.load(config.ckeck_point_dir + load_name, map_location=device)
        model = ckpt['model']
        masks = ckpt['masks']
        train_.set_task(task_index= task_idx-1)
        test_.set_task(task_index = task_idx-1)
        
    train_loader = torch.utils.data.DataLoader(train_, batch_size = 32, shuffle = True, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(test_, batch_size = 32, shuffle = False, num_workers=4)

    model.ti_add_dataset(dataset_list[task_idx-1], n_cls)
    model.ti_set_dataset(dataset_list[task_idx-1])
    model.current_task_index = task_idx
    
    if task_idx == 2: 
        model.hypernetwork_init_classifiers()

    print('[main.py] classifiers', model.classifiers)
    print('[main.py] current classifier', model.classifier)
    kwargs = {
            'optimizer': optim.SGD(
                                [{'params': [p for n, p in model.named_parameters() if 'hypernetwork' not in n]}], 
                                lr = 0.01),
            
            'hyper_optimizer': optim.Adam(
                                [{'params': [p for n, p in model.named_parameters() if 'hypernetwork' in n]}], 
                                lr = 0.001),                    
            'criterion': nn.CrossEntropyLoss(),
            'epochs': config.epochs,
            'pruning_ratio': 0.75,
            'device' : config.device}
    
    manager = Manager(model, masks, train_loader, test_loader, task_idx, **kwargs)
    if task_idx == 2: 
        manager.init_weight('hypernetwork', 0)
    if task_idx >= 2: 
        manager.allocator.initialize_new_mask()
    
    if task_idx >= 3: 
        prev_layer_tokens, generated_weights = manager.model.regul_get_prev_layer_tokens(config.ckeck_point_dir)
    
    if task_idx == 1: 
        manager.epochs = 15

    # get generated weights 
    if task_idx >= 2: 
        manager.epochs = 10
        manager.do_train(get_param=True)
        g_w = manager.model.get_generated_weights()
        
        # train based on generated weights 
        hypernetwork_param, embedding_list = manager.model.hypernetwork.state_dict(), manager.model.embedding_list

        # parameters of embedding list and hypernetwork
        ckpt = torch.load(config.ckeck_point_dir + load_name, map_location=device)
        model = ckpt['model']
        masks = ckpt['masks']
        model.ti_add_dataset(dataset_list[task_idx-1], n_cls)
        model.ti_set_dataset(dataset_list[task_idx-1])
        model.current_task_index = task_idx

        kwargs = {
        'optimizer': optim.SGD(
                            [{'params': [p for n, p in model.named_parameters() if 'hypernetwork' not in n]}], 
                            lr = 0.01),
        
        'hyper_optimizer': optim.Adam(
                            [{'params': [p for n, p in model.named_parameters() if 'hypernetwork' in n]}], 
                            lr = 0.001),                    
        'criterion': nn.CrossEntropyLoss(),
        'epochs': config.epochs,
        'pruning_ratio': 0.75,
        'device' : config.device}

        if task_idx == 2: 
            model.hypernetwork_init_classifiers()
        model.hypernetwork.load_state_dict(hypernetwork_param)
        model.embedding_list = embedding_list
        
        manager = Manager(model, masks, train_loader, test_loader, task_idx, **kwargs)
        manager.allocator.initialize_new_mask()
        manager.generated_parameters = g_w
        manager.epochs = 50

    manager.do_train(get_param=False)
    manager.prune(config.ckeck_point_dir + config.dataset_list[task_idx-1])
    
    if task_idx >=3: 
        manager.do_epoch_regul(100, prev_layer_tokens, generated_weights)
    
    acc, _  = manager.do_eval(task_idx)
    task_incremental_acc[str(current_task)] = acc
    print(task_incremental_acc)
    print('###'*15)