import torch
import torch.nn as nn
from torch.autograd import Function

class MyGelu(Function):
	@staticmethod
	def forward(ctx, input, add_num):
		return nn.GELU()(input) + add_num

	@staticmethod
	def symbolic(g, input, add):
		return g.op("MyGelu", input, add_num_f=add)

mygelu_ = MyGelu.apply

class TinyNet(nn.Module):
	def __init__(self):
		super(TinyNet, self).__init__()

	def forward(self, x):
		x = mygelu_(x, 0.2)
		##x = mygelu_(x, 1.5)
		return x

model = TinyNet().cuda()
ipt = torch.ones(2,3,12,12).cuda()
torch.onnx.export(model, (ipt), 'fuckNet.onnx', opset_version=13, input_names=["input1",], output_names=["outputTensor"],
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
##torch.onnx.export(net, (ipt,), 'tinynet2.onnx', opset_version=11)
##print(onnx.load('tinynet.onnx'))
