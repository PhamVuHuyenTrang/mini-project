
from copy import deepcopy
import torch
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from datasets import get_dataset
from utils.lipschitz import RobustnessOptimizer, add_regularization_args
from sklearn.cluster import KMeans
import numpy as np

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='ER-ACE with future not fixed (as made by authors)'
                                        'Treated with Lipschitz constraints!')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)
    add_regularization_args(parser)

    return parser


class ErACELipschitz(RobustnessOptimizer):
    NAME = 'er_ace_lipschitz'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErACELipschitz, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0

    def begin_task(self, dataset):
        if self.task == 0:
            self.load_initial_checkpoint()
            self.reset_classifier()

            self.net.set_return_prerelu(True)

            self.init_net(dataset)
        
    def end_task(self, dataset):
        self.task += 1

    def to(self, device):
        super().to(device)
        self.seen_so_far = self.seen_so_far.to(device)
    
    def max_pairwise_difference(array_list):
        max_differences = []

        for array in array_list:
            max_difference = np.abs(array - array[:, None]).max()
            max_differences.append(max_difference)

        return max_differences
    

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, epoch=None):
        labels = labels.long()
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)

        loss_re = torch.tensor(0.)
        if self.task > 0:
            # sample from buffer
            if self.task <= 5:

                kmeans = KMeans(n_clusters=self.task + 1, random_state=0, n_init="auto").fit(self.buffer.examples)
    
            else:

                kmeans = KMeans(n_clusters = 5, random_state = 0, n_init = "auto").fit(self.buffer.examples)

            def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
                return np.where(labels_array == clustNum)[0]
            
            #def ClusterIndicesComp(clustNum, labels_array): #list comprehension
            #return np.array([i for i, x in enumerate(labels_array) if x == clustNum])

            mem_buffer_clustered = []
            for i in range(self.task + 1):
                mem_buffer_clustered.append(ClusterIndicesNumpy(i, kmeans.labels_))
    
            self.buffer = np.array(mem_buffer_clustered)

            buf_inputs, buf_labels = self.buffer.get_data(
                self.setting.minibatch_size, transform=self.transform)
            loss_re = self.loss(self.net(buf_inputs), buf_labels)

        if not self.buffer.is_empty():
            if self.args.linear_reg != 0:
                buf_inputs, _ = self.buffer.get_data(self.setting.minibatch_size, transform=self.transform)
                _, buf_output_features = self.net(buf_inputs, returnt='full')
                for output_feature in buf_output_features:
                    loss += self.max_pairwise_difference(output_feature)

            
            if self.args.budget_lip_lambda != 0:
                buf_inputs, _ = self.buffer.get_data(self.setting.minibatch_size, transform=self.transform)
                _, buf_output_features = self.net(buf_inputs, returnt='full')
                         
                if self.args.linear_reg != 0:
                    for single_loss_re in loss_re:
                        loss += self.max_pairwise_difference(single_loss_re)
            

        loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return loss.item(), 0, 0, 0, 0