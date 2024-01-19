from ops import range_view_op, grid2points_op, points2grid_op
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

class Timer:
    def __init__(self, op_name):
        self.begin_time = 0
        self.end_time = 0
        self.op_name = op_name

    def __enter__(self):
        torch.cuda.synchronize()
        self.begin_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        print(f"Average time cost of {self.op_name} is {(self.end_time - self.begin_time) * 1000:.4f} ms")

if __name__ == '__main__':
    n_points = 100
    import pdb;pdb.set_trace()
    with open('./pcd.npy', 'rb') as f_pcd:
        pcd_data = np.load(f_pcd)[ :n_points]
    # import pdb;pdb.set_trace()
    distance = np.sqrt(np.square(pcd_data[:,0]) + np.square(pcd_data[:,1])).reshape(pcd_data.shape[0], 1)
    pcd_data = torch.cat((torch.tensor(pcd_data), torch.tensor(distance)), dim=1)
    points_num = pcd_data.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    points = pcd_data[:,:3].to(device).reshape(-1)
    coors = torch.ones(points_num * 2, dtype=torch.int, device=device, requires_grad=False)
    ##points = torch.ones(n * 3, dtype=torch.float32, device=device, requires_grad=True)
    ##coors = torch.ones(n * 3, dtype=torch.int, device=device, requires_grad=False)
    """
    for point_i in range(pcd_data.shape[0]):
        degree_v = np.arcsin(pcd_data[point_i][2] / np.sqrt(pcd_data[point_i][1] * pcd_data[point_i][1] + pcd_data[point_i][0] * pcd_data[point_i][0] + pcd_data[point_i][2] * pcd_data[point_i][2])) * 180 / 3.14159;
        if (degree_v + 50) / 0.2 > 500:
            import pdb;pdb.set_trace()
            print(degree_v)
    """
    with Timer("range_view"):
        ans = range_view_op(points, coors)
    coors = ans[1].reshape(-1, 2)
    canvas = torch.zeros(
            pcd_data.shape[1],
            3600*500,
            dtype=pcd_data.dtype,
            device=device)
    import pdb;pdb.set_trace()
    points_f = torch.zeros_like(pcd_data).to(device)
    indices = coors[:, 0] * 500 + coors[:, 1]
    indices = indices.long()
    canvas[:, indices] = pcd_data.t().to(device)
    canvas = canvas.view(1, pcd_data.shape[1], 3600, 500)
    RV_image = canvas[0, 3].cpu() * 255
    plt.imsave('example_save_intensity.jpg', RV_image)
    canvas_review = canvas.permute(0, 2, 3, 1).view(-1, pcd_data.shape[1])
    import pdb;pdb.set_trace()
    grid2points_op(points_f, indices, canvas_review)
    print("congradulation!!!")
