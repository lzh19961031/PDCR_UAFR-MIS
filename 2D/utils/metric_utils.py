import torch
import torch.distributed as dist
import numpy as np

EPS = 1e-10

class Evaluator(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.num_class = cfgs.model.num_classes
        self.dataset_len = cfgs.val_len
        self.collection_matrix = []
        self.map_dict = {'mIoU': self.Mean_Intersection_over_Union, 'Acc': self.Pixel_Accuracy, 'mDice': self.Mean_Dice_Coefficient}

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU
    
    def Mean_Dice_Coefficient(self):
        A_inter_B = np.diag(self.confusion_matrix)
        A = np.sum(self.confusion_matrix, axis=1)
        B = np.sum(self.confusion_matrix, axis=0)

        dice = (2 * A_inter_B) / (A + B + EPS)
        avg_dice = self.nanmean(dice)
        return avg_dice


    def nanmean(self, x):
        return np.mean(x[x == x])

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = np.reshape(count, (self.num_class, self.num_class))
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        if torch.is_tensor(gt_image):
            gt_image = gt_image.cpu().numpy()
        if torch.is_tensor(pre_image):
            pre_image = pre_image.cpu().numpy()
        assert gt_image.shape == pre_image.shape, print('shape not match!')
        confusion_matrix = []
        for i in range(gt_image.shape[0]):
            confusion_matrix.append(self._generate_matrix(gt_image[i, :], pre_image[i, :]))
        self.collection_matrix.append(np.stack(confusion_matrix, axis=0))

    def distributed_concat(self):
        self.collection_matrix = np.concatenate(self.collection_matrix, axis=0)
        tensor = torch.tensor(self.collection_matrix).cuda()
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        self.all_confusion_matrix = torch.cat(output_tensors, dim=0)[:self.dataset_len, :]
        self.all_confusion_matrix = self.all_confusion_matrix.cpu().numpy()

    def get_metric(self):
        result_dict = {}
        for metric_name in self.cfgs.val.metric_used:
            result_dict[metric_name] = []

        for i in range(self.all_confusion_matrix.shape[0]):
            self.confusion_matrix = self.all_confusion_matrix[i, :]
            for metric_name in self.cfgs.val.metric_used:
                result_dict[metric_name].append(np.around(self.map_dict[metric_name](), 4))

        for key in result_dict.keys():
            result_dict[key] = np.mean(result_dict[key])
    
        return result_dict



