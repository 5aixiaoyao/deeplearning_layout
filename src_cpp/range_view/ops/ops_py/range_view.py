import torch
from torch.autograd import Function
import range_view
import grid2points
import points2grid

class RangeView(Function):

    @staticmethod
    def forward(ctx, points, coors):
        points = points.float()
        coors = coors.int()
        range_view.forward(points.contiguous(), coors.contiguous())

        ctx.mark_non_differentiable(coors)
        ctx.mark_non_differentiable(points)

        return points, coors

    @staticmethod
    def backward(ctx, g_out):
        return None
    
class Grid2Points(Function):

    @staticmethod
    def forward(ctx, points_f, indices, grid_f):
        points_f = points_f.float()
        indices = indices.int()
        grid_f = grid_f.float()
        # import pdb;pdb.set_trace()
        grid2points.forward(points_f.contiguous(), indices.contiguous(), grid_f.contiguous())

        ctx.mark_non_differentiable(points_f)
        ctx.mark_non_differentiable(indices)
        ctx.mark_non_differentiable(grid_f)

        return points_f

    @staticmethod
    def backward(ctx, g_out):
        return None

class Points2Grid(Function):

    @staticmethod
    def forward(ctx, points_f, indices, grid_f):
        points_f = points_f.float()
        indices = indices.int()
        grid_f = grid_f.float()
        # import pdb;pdb.set_trace()
        points2grid.forward(points_f.contiguous(), indices.contiguous(), grid_f.contiguous())

        ctx.mark_non_differentiable(points_f)
        ctx.mark_non_differentiable(indices)
        ctx.mark_non_differentiable(grid_f)

        return points_f

    @staticmethod
    def backward(ctx, g_out):
        return None

range_view_op = RangeView.apply
grid2points_op = Grid2Points.apply
points2grid_op = Points2Grid.apply
