import torch
import torch.nn as nn
from torch.autograd import Function
import onnx
import torch.onnx
from onnx2engine_py import onnx2Engine, test_engine
import numpy as np

class Requant_(Function):
    @staticmethod
    def forward(ctx, input, requant_scale, shift):
        input = input.double() * requant_scale / 2 ** shift
        input = torch.floor(input).float()
        
        return torch.floor(input)

    @staticmethod
    def symbolic(g, *inputs):
        return g.op("PillarScatter", inputs[0], grid_x_i=23, grid_y_i=23, num_features_i=2)
        #return g.op("RequantPlugin", inputs[0], scale_f=23.0, shift_i=8)

requant_ = Requant_.apply

class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(-1)
        x = requant_(x, 5, 5)
        return x

net = TinyNet().cuda()
ipt = torch.ones(1, 3, 12, 12).cuda()
result = net(ipt)
torch.onnx.export(net, (ipt), 'TinyNet.onnx', opset_version=13, input_names=["input1",], output_names=["outputTensor"],
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
engine_trt = onnx2Engine("./centerpoint_m_199_0103.onnx", "Centerpoint.engine", [3, 12, 12], 5 * (1<<30), 1)
import pdb;pdb.set_trace()
outputs = test_engine("Tinyengine.engine", ipt)
import pdb;pdb.set_trace()
output_cpu = np.frombuffer(outputs[0], dtype= np.float32).reshape(1, 1, 12, 12)
##print(output_cpu)

##print(onnx.load('tinynet.onnx'))
