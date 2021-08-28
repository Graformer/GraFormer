import torch.nn as nn
import numpy as np
import torch

class nddr_layer(nn.Module):
    def __init__(self, in_channels, out_channels, task_num, init_weights=[0.9, 0.1], init_method='constant'):
        super(nddr_layer, self).__init__()
        self.task_num = task_num
        assert task_num>=2, 'Task Num Must >=2'

        self.Conv_Task_List = nn.ModuleList([])
        for i in range(self.task_num):
            task_basic = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, 1, 0),
                                  #     nn.BatchNorm2d(out_channels),
                                       nn.ReLU(True))
            self.Conv_Task_List.append(task_basic)

        self_w,  others_w = init_weights[0],  init_weights[1]/(self.task_num-1)
        others_diag = others_w * np.diag(np.ones(out_channels)).astype(dtype=np.float32)
        self_w_diag = self_w * np.diag(np.ones(out_channels))

        if init_method == 'constant':
            for i in range(self.task_num):
                diag_M = np.tile(others_diag, (1, self.task_num))
                print('diag m', diag_M.shape)
                start_id = int(i*out_channels)
                end_id = start_id + out_channels
                diag_M[:, start_id:end_id] = self_w_diag

                self.Conv_Task_List[i][0].weight = torch.nn.Parameter(torch.from_numpy(diag_M[:, :, np.newaxis, np.newaxis]))
                print('conv task i 0 weight', self.Conv_Task_List[i][0].weight)
                torch.nn.init.constant_(self.Conv_Task_List[i][0].bias, 0.0)

    def forward(self, Net_F):
        Net_Res = []
        for i in range(self.task_num):
            print('net f shape', Net_F.shape)
            tmp = self.Conv_Task_List[i](Net_F)
            Net_Res.append(tmp)

        return Net_Res


if __name__ == '__main__':
    nddr = nddr_layer(in_channels=2 * 2, out_channels=2, task_num=2)
    x = torch.ones(1, 2, 4) * 3
    y = torch.ones(1, 2, 4) * 7
    rs = nddr(torch.cat((x, y), dim=1))
    print(rs)


