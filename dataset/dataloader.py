import torch.utils.data as data
import torchvision

from .transforms import *

from PIL import Image
import os
import os.path
import numpy as np
import torch
from numpy.random import randint

import pickle

'''
video_list = [[path, num_frames, label], [path, num_frames, label], ...]
'''

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file, task_index, 
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, mode='train'):
        
        ''' Load CIL Task Config Files'''
        with open(list_file, 'rb') as f:
            self.list_file = pickle.load(f)

        self.dataset_name = list_file.split('/')[-1].split('_')[0].lower()
        
        self.root_path = root_path
        if self.dataset_name == 'Kinetics400'.lower():
            self.root_path = root_path + '/' + mode.lower()
            
            ''' Load Missing File List'''
            self.missing_list = []
            with open(f'./vClimb-missing/missing_{mode}.txt', 'r') as f:
                missing_list = f.readlines()

            for line in missing_list:
                self.missing_list.append(line[:-1])

            #print(self.missing_list)
            #quit()

        else:
            self.missing_list = None

        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform        
        
        self.random_shift = random_shift

        self.mode = mode.lower()
        assert self.mode in ['train', 'val', 'test'], '[!] mode argument should be train, val or test.'
        assert not (self.mode == 'test') or not (self.dataset_name == 'activitynet'), '[!] ActivityNet does not have test set.'
        
        ''' Set Task '''
        self.curr_task_index = task_index
        self._get_video_list(self.list_file, task_index)
    
    def set_task(self, task_index):
        self.curr_task_index = task_index
        self._get_video_list(self.list_file, task_index)


    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """

        # average_duration = (record[1] - self.new_length + 1) // self.num_segments
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        # elif record[1] > self.num_segments:
        elif record.num_frames > self.num_segments:
            # offsets = np.sort(randint(record[1] - self.new_length + 1, size=self.num_segments))
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets 

    def _get_val_indices(self, record):
        # if record[1] > self.num_segments + self.new_length - 1:
        if record.num_frames  > self.num_segments + self.new_length - 1:
            # tick = (record[1] - self.new_length + 1) / float(self.num_segments)
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets 

    def _get_test_indices(self, record):

        # tick = (record[1] - self.new_length + 1) / float(self.num_segments)
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets 

    def _get_num_frames(self, video):
        # root path : /disk1/UCF101
        # videos = data_dict['train'][task_idx]['ApplyEyeMakeup']
        # videos = [video1, ... video N] 
        video = os.path.join(self.root_path, video)
        return len(os.listdir(video))
    
    def _get_video_list(self, dataset_dict, task_idx):
        # Make path, number of frames, and label 
        self.video_list = []

        labels = list(dataset_dict[self.mode][task_idx].keys())
        # make labels 
        labels2index = {}
        for index, label in enumerate(labels):
            labels2index[label] = index
        
        # make path 
        for label in labels:
            for video in dataset_dict[self.mode][task_idx][label]:
                if self.dataset_name == 'ActivityNet'.lower():
                    video = video['filename']
                
                if self.dataset_name == 'ucf101':
                    if 'HandStandPushups' in video: 
                        video = video.replace('HandStandPushups', 'HandstandPushups')

                if self.dataset_name == 'Kinetics400':
                    if video in self.missing_list:
                        continue
                self.video_list.append(VideoRecord([os.path.join(self.root_path, video), self._get_num_frames(video), labels2index[label]]))

    def __getitem__(self, index):
        record = self.video_list[index]

        if self.mode == 'train':
            segment_indices = self._sample_indices(record) # if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                # seg_imgs = self._load_image(record[0], p)
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                # if p < record[1]:
                if p < record.num_frames:
                    p += 1

        process_data = self.transform((images))
        # return process_data, record[2]
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)



if __name__ == '__main__':

    ttransform=torchvision.transforms.Compose([
                        GroupMultiScaleCrop(224, [1, .875, .75, .66]),
                        GroupRandomHorizontalFlip(is_flow=False),
                        Stack(roll = False),
                        ToTorchFormatTensor(div=True),
                   ])


    # pickle_file = './vClimb-dataset/UCF101_data_10cls_10tasks.pkl'
    # data = TSNDataSet('/workspace/data/UCF101/rgb', pickle_file, task_index=0, num_segments=3, new_length=1, modality='RGB', image_tmpl='img_{:05d}.jpg', transform=ttransform, force_grayscale=False, random_shift=True, mode='test')
    
    # pickle_file = './vClimb-dataset/ActivityNet_data_20cls_10tasks.pkl'
    # data = TSNDataSet('/workspace/data/ActivityNet/Activitynet-200/rgb', pickle_file, task_index=2, num_segments=3, new_length=1, modality='RGB', image_tmpl='img_{:05d}.jpg', transform=ttransform, force_grayscale=False, random_shift=True, mode='test')

    pickle_file = './vClimb-dataset/Kinetics400_data_40cls_10tasks.pkl'
    data = TSNDataSet('/workspace/video_dataset/kinetics/k400_rgb', pickle_file, task_index=0, num_segments=3, new_length=1, modality='RGB', image_tmpl='img_{:05d}.jpg', transform=ttransform, force_grayscale=False, random_shift=True, mode='train')

    accumulated_cnt = 0
    for task_index in range(10):
        task_cnt = 0
        print('='*20 + f' Task {task_index+1} ' + '='*20)
        
        if task_index != 0:
            data.set_task(task_index)

        data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    
        for i, x in enumerate(data_loader):
            accumulated_cnt += len(x[0])
            task_cnt += len(x[0])
            print(i+1, task_cnt, accumulated_cnt)

    #x = next(iter(data_loader))
    #print(len(x))
    #print(x[0])
    #print(x[1])
    #print(len(data.video_list))    
