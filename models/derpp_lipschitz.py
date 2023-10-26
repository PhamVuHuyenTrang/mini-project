from datasets import get_dataset
from utils.buffer import Buffer
from torch.nn import functional as F
from utils.args import *
import torch
import numpy as np

from utils.lipschitz import RobustnessOptimizer, add_regularization_args
from sklearn.cluster import KMeans
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.'
                                        'Treated with Lipschitz constraints!')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)
    add_regularization_args(parser)

    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')

    return parser

class DerppLipschitz(RobustnessOptimizer):
    NAME = 'derpp_lipschitz'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DerppLipschitz, self).__init__(backbone, loss, args, transform)

        if args.distributed != 'ddp':
            self.buffer = Buffer(self.args.buffer_size, self.device)
        else:
            import os
            partial_buf_size = self.args.buffer_size // int(os.environ['MAMMOTH_WORLD_SIZE'])

            print('using partial buf size', partial_buf_size)
            self.buffer = Buffer(partial_buf_size, self.device)

        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        self.n_tasks = get_dataset(args).N_TASKS

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.load_initial_checkpoint()
            self.reset_classifier()
                                
            self.net.set_return_prerelu(True)
            
            self.init_net(dataset)

    def end_task(self, dataset):
        self.current_task += 1
    def max_pairwise_difference(array_list):
        max_differences = []

        for array in array_list:
            max_difference = np.abs(array - array[:, None]).max()
            max_differences.append(max_difference)

        return max_differences

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, epoch=None):
        labels = labels.long()
        self.opt.zero_grad()

        outputs = self.net(inputs)

        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            if self.current_task <= 5:

                kmeans = KMeans(n_clusters=self.current_task + 1, random_state=0, n_init="auto").fit(self.buffer.examples)
    
            else:

                kmeans = KMeans(n_clusters = 5, random_state = 0, n_init = "auto").fit(self.buffer.examples)

            def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
                return np.where(labels_array == clustNum)[0]

            #def ClusterIndicesComp(clustNum, labels_array): #list comprehension
                #return np.array([i for i, x in enumerate(labels_array) if x == clustNum])

            mem_buffer_clustered = []
            for i in range(self.current_task + 1):
                mem_buffer_clustered.append(ClusterIndicesNumpy(i, kmeans.labels_))
    
            self.buffer = np.array(mem_buffer_clustered)
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.setting.minibatch_size, transform=self.transform)
            buf_outputs, buf_output_features = self.net(buf_inputs, returnt='full')
            loss_1 = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss += loss_1

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.setting.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs).float()
            loss_2 = self.args.beta * self.loss(buf_outputs, buf_labels)
            loss += loss_2
        
            if self.args.linear_reg != 0:
                buf_inputs, _, _ = self.buffer.get_data(self.setting.minibatch_size, transform=self.transform)
                _, buf_output_features = self.net(buf_inputs, returnt='full')

                for output_feature in buf_output_features:
                    loss += self.max_pairwise_difference(output_feature)
            
            if self.args.loss_reg != 0:
                buf_inputs, _, _ = self.buffer.get_data(self.setting.minibatch_size, transform=self.transform)
                _, buf_output_features = self.net(buf_inputs, returnt='full')

                for single_loss in loss:
                    loss += self.max_pairwise_difference(single_loss)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item(),0,0,0,0
