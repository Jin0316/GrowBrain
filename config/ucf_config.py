from dataset.transforms import *

device = 'cuda:0'
dataset_list = []
n_task = 10 

for task in range(n_task):
    dataset_list.append('ucf' + str(task))

dataset_classes = [11, 10, 10, 10, 10, 10, 10, 10, 10, 10]

save_names = []
for i in range(len(dataset_list)):
    save_names.append(dataset_list[i] + '.pt')

ckeck_point_dir = './check_point/'

root_path = '/{path_to_dataset}'
transform=torchvision.transforms.Compose([
                        GroupMultiScaleCrop(224, [1, .875, .75, .66]),
                        GroupRandomHorizontalFlip(is_flow=False),
                       Stack(roll = False),
                       ToTorchFormatTensor(div=True),
                   ])

data_name = 'ucf'
method = 'GrowBrain'
basemodel = 'resnet34'
num_segments = 3 
epochs = 50
lambda_sp = 0.1
modality = 'RGB'
pickle_file = './pkl_files/UCF101_data.pkl'