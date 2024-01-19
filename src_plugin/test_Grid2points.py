import torch
import torch.nn as nn
from torch.autograd import Function
import onnx
import torch.onnx
from onnx2engine_py import onnx2Engine, test_engine
import numpy as np

class Grid2Points(Function):
    @staticmethod
    def forward(ctx, indice, rv_x_i, points_num):
        #grid2points_op(points_rv, indice, rv_x_i)
        points_rv = torch.ones(1, 4000, 64).cuda()
        return torch.floor(points_rv)

    @staticmethod
    def symbolic(g, indice, rv_x_i, points_num):
        return g.op("Grid2PointsPlugin", points_num, indice, rv_x_i, channels_i=64, features_num_i=450000)

grid2points = Grid2Points.apply

class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
    def forward(self, rv_coors, rv_x_i, points_num):
        x = self.relu1(rv_x_i)
        points_rv = grid2points(rv_coors, x, points_num)
        return points_rv

net = TinyNet().cuda()
points_rv = torch.ones(1, 4000, 64).cuda()
rv_coors = torch.ones(1,1,1,4000).cuda().int()
rv_x_i = torch.ones(1, 4500, 64).cuda()
points_num = torch.tensor([[[[4000]]]]).cuda().int()
result = net(rv_coors, rv_x_i, points_num)
torch.onnx.export(net, (rv_coors, rv_x_i, points_num), 'Grid2Points.onnx', opset_version=11, input_names=["rv_coors"," rv_x_i", "points_num"], output_names=["outputTensor"],
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
engine_trt = onnx2Engine("./Grid2Points.onnx", "Grid2Points.engine", [3, 12, 12], 50 * (1<<30), 1)
import pdb;pdb.set_trace()
outputs = test_engine("Grid2Points.engine", points_rv)
import pdb;pdb.set_trace()
output_cpu = np.frombuffer(outputs[0], dtype= np.float32).reshape(1, 1, 12, 12)
##print(output_cpu)

##print(onnx.load('tinynet.onnx'))

