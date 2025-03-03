import torch
import torch.nn as nn


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, task1_loss, task2_loss, task3_loss=None):

        # 对于第一个任务，使用传入的损失张量
        precision1 = torch.exp(-self.log_vars[0])
        loss = precision1 * task1_loss + self.log_vars[0]

        # 对于第二个任务，使用传入的损失张量
        precision2 = torch.exp(-self.log_vars[1])
        loss += precision2 * task2_loss + self.log_vars[1]

        # precision3 = torch.exp(-self.log_vars[2])
        # loss += precision3 * task3_loss + self.log_vars[2]

        loss = torch.mean(loss)

        return loss, self.log_vars.data.tolist()
