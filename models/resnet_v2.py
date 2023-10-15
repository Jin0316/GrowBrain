import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityLayer(nn.Module):
    def forward(self, x): 
        return x 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    """
    conv1: 3 x 3 conv1 in original basic block
    conv2: 3 x 3 conv2 in original basic block
    conv3: 1 x 1 shortcut in original basic block 
    """

    def forward(self, x, g_conv1, g_conv2, g_conv3): 
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class no_downsample_BasicBlock(nn.Module):
    expansion = 1 
    def __init__(self, in_planes, planes, stride = 1):
        super(no_downsample_BasicBlock, self).__init__()
        self.no_hyper_mode = False
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(self.planes)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.conv1 = nn.parameter.Parameter(torch.randn(planes,in_planes,3,3), requires_grad=True)
        self.conv2 = nn.parameter.Parameter(torch.randn(planes,planes,3,3), requires_grad=True)
        self.identity_layer = IdentityLayer()
        self.g_conv1, self.g_conv2 = None, None 

    def forward(self, x, g_conv1, g_conv2): 
        if self.no_hyper_mode == False:
            self.g_conv1, self.g_conv2 = g_conv1.detach(), g_conv2.detach()
            out = F.relu(self.bn1(F.conv2d(x, self.conv1*g_conv1, stride = self.stride, padding = 1)))
            out = self.bn2(F.conv2d(out, self.conv2*g_conv2, stride = 1, padding = 1))
            out += self.identity_layer(x)
            out = F.relu(out)
            return out 

        elif self.no_hyper_mode == True:
            out = F.relu(self.bn1(F.conv2d(x, self.conv1, stride = self.stride, padding = 1)))
            out = self.bn2(F.conv2d(out, self.conv2, stride = 1, padding = 1))
            out += self.identity_layer(x)
            out = F.relu(out)
            return out 


class downsample_BasicBlock(nn.Module):
    expansion = 1 
    def __init__(self, in_planes, planes, stride = 1):
        super(downsample_BasicBlock, self).__init__()
        self.no_hyper_mode = False
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(self.planes)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.bn3 = nn.BatchNorm2d(self.planes)

        self.conv1 = nn.parameter.Parameter(torch.randn(planes, in_planes, 3, 3), requires_grad=True)
        self.conv2 = nn.parameter.Parameter(torch.randn(planes, planes, 3, 3), requires_grad=True)
        self.conv3 = nn.parameter.Parameter(torch.randn(planes, in_planes, 1, 1), requires_grad=True)
        self.g_conv1, self.g_conv2, self.g_conv3 = None, None, None 

    def forward(self, x, g_conv1, g_conv2, g_conv3):
        if self.no_hyper_mode == False:
            #print('resnet downsample block | no hyper false')
            self.g_conv1, self.g_conv2, self.g_conv3 = g_conv1.detach(), g_conv2.detach(), g_conv3.detach()
            out = F.relu(self.bn1(F.conv2d(x, self.conv1*g_conv1, stride = self.stride, padding = 1)))
            out = self.bn2(F.conv2d(out, self.conv2*g_conv2, stride = 1, padding = 1))
            shortcut = self.bn3(F.conv2d(x, self.conv3*g_conv3, stride = self.stride, padding = 0 ))
            out += shortcut 
            out = F.relu(out)
            return out 

        elif self.no_hyper_mode == True:
            #print('resnet downsample block | no hyper true')
            out = F.relu(self.bn1(F.conv2d(x, self.conv1, stride = self.stride, padding = 1)))
            out = self.bn2(F.conv2d(out, self.conv2, stride = 1, padding = 1))
            shortcut = self.bn3(F.conv2d(x, self.conv3, stride = self.stride, padding = 0 ))
            out += shortcut
            out = F.relu(out)
            return out


class nodownsample_BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, stride = 1):
        super(nodownsample_BottleneckBlock, self).__init__()
        expansion = 4 
        self.no_hyper_mode = False
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(self.planes)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.bn3 = nn.BatchNorm2d(expansion * self.planes)

        self.conv1 = nn.parameter.Parameter(torch.randn(planes, in_planes, 1, 1), requires_grad=True)
        self.conv2 = nn.parameter.Parameter(torch.randn(planes, planes, 3, 3), requires_grad=True)
        self.conv3 = nn.parameter.Parameter(torch.randn(expansion * planes, planes, 1, 1), requires_grad=True)
        self.g_conv1, self.g_conv2, self.g_conv3 = None, None, None 
        self.identity_layer = IdentityLayer()

    def forward(self, x, g_conv1,  g_conv2,  g_conv3):
        if self.no_hyper_mode == False:
            self.g_conv1, self.g_conv2, self.g_conv3 = g_conv1.detach(), g_conv2.detach(), g_conv3.detach()
            out = F.relu(self.bn1(F.conv2d(x,   self.conv1 * g_conv1, stride = 1, padding = 1)))
            out = F.relu(self.bn2(F.conv2d(out, self.conv2 * g_conv2, stride = self.stride, padding = 1)))
            out = self.bn3(F.conv2d(out, self.conv3 * g_conv3, stride = 1, padding = 1))
            out += self.identity_layer(x)
            out = F.relu(out)
            return out 

        elif self.no_hyper_mode == True: 
            out = F.relu(self.bn1(F.conv2d(x,   self.conv1, stride = 1, padding = 1)))
            out = F.relu(self.bn2(F.conv2d(out, self.conv2, stride = self.stride, padding = 1)))
            out = self.bn3(F.conv2d(out, self.conv3, stride = 1, padding = 1))
            out += self.identity_layer(x)
            out = F.relu(out)
            return out

class downsample_BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, stride = 1):
        super(downsample_BottleneckBlock, self).__init__()
        expansion = 4
        self.no_hyper_mode = False
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(self.planes)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.bn3 = nn.BatchNorm2d(expansion * self.planes)
        self.bn4 = nn.BatchNorm2d(expansion * self.planes)

        self.conv1 = nn.parameter.Parameter(torch.randn(planes, in_planes, 1, 1), requires_grad=True)
        self.conv2 = nn.parameter.Parameter(torch.randn(planes, planes, 3, 3), requires_grad=True)
        self.conv3 = nn.parameter.Parameter(torch.randn(expansion * planes, planes, 1, 1), requires_grad=True)
        self.conv4 = nn.parameter.Parameter(torch.randn(expansion * planes, in_planes, 1, 1), requires_grad=True)
        self.g_conv1, self.g_conv2, self.g_conv3, self.g_conv4 = None, None, None, None

    def forward(self, x, g_conv1, g_conv2, g_conv3, g_conv4): 
        if self.no_hyper_mode == False:
            self.g_conv1, self.g_conv2, self.g_conv3, self.g_conv4 = g_conv1.detach(), g_conv2.detach(), g_conv3.detach(), g_conv4.detach()
            out = F.relu(self.bn1(F.conv2d(x, self.conv1 * g_conv1, stride = 1, padding = 0)))
            out = F.relu(self.bn2(F.conv2d(out, self.conv2 * g_conv2, stride = self.stride, padding = 1)))
            out = self.bn3(F.conv2d(out, self.conv3 * g_conv3, stride = 1, padding = 0))
            out += self.bn4(F.conv2d(x, self.conv4 * g_conv4, stride = self.stride))
            out = F.relu(out)
            return out 

        elif self.no_hyper_mode == True: 
            out = F.relu(self.bn1(F.conv2d(x, self.conv1, stride = 1, padding = 0)))
            out = F.relu(self.bn2(F.conv2d(out, self.conv2, stride = self.stride, padding = 1)))
            out = self.bn3(F.conv2d(out, self.conv3, stride = 1, padding = 0))
            out += self.bn4(F.conv2d(x, self.conv4, stride = self.stride))
            out = F.relu(out)
            return out 


class nodownsample_BottleneckBlock_v2(nn.Module):
    def __init__(self, in_planes, planes, stride = 1):
        super(nodownsample_BottleneckBlock_v2, self).__init__()
        expansion = 4 
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = nn.parameter.Parameter(torch.randn(self.planes, self.in_planes, 1, 1), requires_grad=True)
        self.bn1 = nn.BatchNorm2d(self.planes)
        self.conv2 = nn.parameter.Parameter(torch.randn(self.planes, self.planes, 3, 3), requires_grad=True)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.conv3 = nn.parameter.Parameter(torch.randn(expansion * self.planes, self.planes, 1, 1), requires_grad=True)
        self.bn3 = nn.BatchNorm2d(expansion * self.planes)
    
        self.identity_layer = IdentityLayer()

    def forward(self, x):
        out = F.relu(self.bn1(F.conv2d(x, self.conv1, stride = 1, padding = 0)))
        out = F.relu(self.bn2(F.conv2d(out, self.conv2, stride = self.stride, padding = 1)))
        out = self.bn3(F.conv2d(out, self.conv3, stride = 1, padding = 0))
        out += self.identity_layer(x)
        out = F.relu(out)
        return out

class downsample_BottleneckBlock_v2(nn.Module):
    def __init__(self, in_planes, planes, stride = 1):
        super(downsample_BottleneckBlock_v2, self).__init__()
        expansion = 4
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(self.planes)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.bn3 = nn.BatchNorm2d(expansion * self.planes)
        self.bn4 = nn.BatchNorm2d(expansion * self.planes)

        self.conv1 = nn.parameter.Parameter(torch.randn(self.planes, self.in_planes, 1, 1), requires_grad=True)
        self.conv2 = nn.parameter.Parameter(torch.randn(self.planes, self.planes, 3, 3), requires_grad=True)
        self.conv3 = nn.parameter.Parameter(torch.randn(expansion * self.planes, self.planes, 1, 1), requires_grad=True)
        self.conv4 = nn.parameter.Parameter(torch.randn(expansion * self.planes, self.in_planes, 1, 1), requires_grad=True)

    def forward(self, x): 
        out = F.relu(self.bn1(F.conv2d(x, self.conv1, stride = 1, padding = 0)))
        out = F.relu(self.bn2(F.conv2d(out, self.conv2, stride = self.stride, padding = 1)))
        out = self.bn3(F.conv2d(out, self.conv3, stride = 1, padding = 0))
        out += self.bn4(F.conv2d(x, self.conv4, stride = self.stride))
        out = F.relu(out)
        return out 