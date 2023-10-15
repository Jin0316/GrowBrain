import torch 
import math 

class Identity(torch.nn.Module):
    def forward(self, input):
        return input 

class SegmentConsensus(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(torch.tensor(input_tensor.size()))
        output = input_tensor.mean(dim = 1, keepdim=True)
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        input_shape = ctx.saved_tensors
        input_shape = input_shape[0].tolist()
        grad_in = grad_output.expand(tuple(input_shape))/float(input_shape[1])
        return grad_in 

class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim).apply(input)


if __name__ == '__main__':
    random_input = torch.randn()