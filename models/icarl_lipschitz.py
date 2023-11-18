
from copy import deepcopy

from functions.augmentations import normalize
import torch
import torch.nn.functional as F
from datasets import get_dataset
from functions.buffer import Buffer
from functions.args import *
from models.utils.continual_model import ContinualModel
from functions.distributed import make_dp
from functions.lipschitz import RobustnessOptimizer, add_regularization_args
from functions.create_partition import create_partition_func_1nn
from functions.no_bn import bn_track_stats
import numpy as np
from functions.augmentations import rotate_30_degrees, rotate_60_degrees, add_noise, change_colors

partition_func = create_partition_func_1nn((84, 84, 3), n_centroids=5000)

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via iCaRL.'
                            'Treated with Lipschitz constraints!')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)
    add_regularization_args(parser)

    parser.add_argument('--wd_reg', type=float, required=True,
                        help='L2 regularization applied to the parameters.')
                        
    return parser

def icarl_fill_buffer(self: ContinualModel, mem_buffer: Buffer, dataset, t_idx: int) -> None:
    """
    Adds examples from the current task to the memory buffer
    by means of the herding strategy.
    :param mem_buffer: the memory buffer
    :param dataset: the dataset from which take the examples
    :param t_idx: the task index
    """

    mode = self.net.training
    self.net.eval()
    samples_per_class = mem_buffer.buffer_size // (dataset.N_CLASSES_PER_TASK * (t_idx + 1))

    if t_idx > 0:
        # 1) First, subsample prior classes
        buf_x, buf_y, buf_l, buf_clusterID = self.buffer.get_all_data()

        mem_buffer.empty()
        for _y in buf_y.unique():
            idx = (buf_y == _y)
            _y_x, _y_y, _y_l, _y_clusterID = buf_x[idx], buf_y[idx], buf_l[idx],buf_clusterID[idx]
            mem_buffer.add_data(
                examples=_y_x[:samples_per_class],
                labels=_y_y[:samples_per_class],
                logits=_y_l[:samples_per_class],
                clusterID = _y_clusterID[:samples_per_class]
            )

    # 2) Then, fill with current tasks
    loader = dataset.train_loader
    mean, std = dataset.get_denormalization_transform().mean, dataset.get_denormalization_transform().std
    classes_start, classes_end = t_idx * dataset.N_CLASSES_PER_TASK, (t_idx+1) * dataset.N_CLASSES_PER_TASK
    # # todo add normalize to features for other datasets

    # 2.1 Extract all features
    a_x, a_y, a_f, a_l = [], [], [], []
    for x, y, not_norm_x in loader:
        mask = (y >= classes_start) & (y < classes_end)
        x, y, not_norm_x = x[mask], y[mask], not_norm_x[mask]
        if not x.size(0):
            continue
        x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
        a_x.append(not_norm_x.to('cpu'))
        a_y.append(y.to('cpu'))
        try:
            feats = self.net.features(normalize(not_norm_x, mean, std)).float()
            outs = self.net.classifier(feats)
        except:
            replica_hack = False
            if self.args.distributed == 'dp' and len(not_norm_x) < torch.cuda.device_count():
                # yup, that's exactly right! I you have more GPUs than inputs AND you use kwargs,
                # dataparallel breaks down. So we pad with mock data and then ignore the padding.
                # ref https://github.com/pytorch/pytorch/issues/31460
                replica_hack = True
                not_norm_x = not_norm_x.repeat(torch.cuda.device_count(), 1, 1, 1)

            outs, feats = self.net(normalize(not_norm_x, mean, std), returnt='both')

            if replica_hack:
                outs, feats = outs.split(len(not_norm_x) // torch.cuda.device_count())[0], feats.split(len(not_norm_x) // torch.cuda.device_count())[0]

        a_f.append(feats.cpu())
        a_l.append(torch.sigmoid(outs).cpu())
    a_x, a_y, a_f, a_l = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f), torch.cat(a_l)

    
    # 2.2 Compute class means
    for _y in range(classes_start, classes_end):
        idx = (a_y == _y)
        _x, _y, _l = a_x[idx], a_y[idx], a_l[idx]
        feats = a_f[idx]
        feats = feats.reshape(len(feats), -1)
        mean_feat = feats.mean(0, keepdim=True)

        running_sum = torch.zeros_like(mean_feat)
        i = 0
        while i < samples_per_class and i < feats.shape[0]:
            cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

            idx_min = cost.argmin().item()

            mem_buffer.add_data(
                examples=_x[idx_min:idx_min + 1].to(self.device),
                labels=_y[idx_min:idx_min + 1].to(self.device),
                logits=_l[idx_min:idx_min + 1].to(self.device),
                clusterID=partition_func(_x[idx_min:idx_min + 1]).to(self.device)

            )

            running_sum += feats[idx_min:idx_min + 1]
            feats[idx_min] = feats[idx_min] + 1e6
            i += 1

    assert len(mem_buffer.examples) <= mem_buffer.buffer_size
    assert mem_buffer.num_seen_examples <= mem_buffer.buffer_size
    self.net.train(mode)


class ICarlLipschitz(RobustnessOptimizer):
    NAME = 'icarl_lipschitz'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ICarlLipschitz, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK *
                             self.dataset.N_TASKS).to(self.device)

        self.class_means = None
        self.icarl_old_net = None
        self.current_task = 0
        self.num_classes = self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS

    def to(self, device):
        self.eye = self.eye.to(device)
        return super().to(device)

    def forward(self, x):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()
                self.class_means = self.class_means.squeeze()

        try:
            feats = self.net.features(x).float().squeeze()
        except:
            feats = self.net(x, returnt='both')[1].float().squeeze()
        
        feats = feats.reshape(feats.shape[0], -1)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, logits=None, epoch=None):
        if not hasattr(self, 'classes_so_far_buffer'):
            self.classes_so_far_buffer = labels.unique().to(labels.device)
            self.register_buffer('classes_so_far', self.classes_so_far_buffer)
        else:
            self.classes_so_far_buffer = torch.cat((self.classes_so_far_buffer, labels.to(self.classes_so_far_buffer.device))).unique()

            self.register_buffer('classes_so_far', self.classes_so_far_buffer)

        self.class_means = None
        if self.current_task > 0:
            with torch.no_grad():
                logits = torch.sigmoid(self.icarl_old_net(inputs))
        self.opt.zero_grad()
        loss, _ = self.get_loss(inputs, labels, self.current_task, logits)
        
        # Robustness losses (New regularization)
        unique_labels = torch.unique(labels)
        num_classes_so_far = unique_labels.numel()


        if not self.buffer.is_empty():
            print("start buffer")
            mean, std = self.dataset.get_denormalization_transform().mean, self.dataset.get_denormalization_transform().std
            buffer_x, buffer_y, buffer_logits, _ = self.buffer.get_all_data()
            
            with torch.no_grad():
                rotate_30_degrees_data = rotate_30_degrees(self.buffer.examples.clone().detach(), mean, std)
                rotate_60_degrees_data = rotate_60_degrees(self.buffer.examples.clone().detach(), mean, std)
                add_noise_data = add_noise(self.buffer.examples.clone().detach(), mean, std)
                change_colors_data = change_colors(self.buffer.examples.clone().detach(), mean, std)
                augment_examples = torch.cat([rotate_30_degrees_data, rotate_60_degrees_data, add_noise_data, change_colors_data], dim=0)


            rotate_30_degrees_logits = torch.sigmoid(self.net(rotate_30_degrees_data))
            rotate_60_degrees_logits = torch.sigmoid(self.net(rotate_60_degrees_data))
            add_noise_logits = torch.sigmoid(self.net(add_noise_data))
            change_colors_logits = torch.sigmoid(self.net(change_colors_data))
            augmented_logits = torch.cat([rotate_30_degrees_logits, rotate_60_degrees_logits, add_noise_logits, change_colors_logits], dim=0)
            augmented_labels = torch.argmax(augmented_logits, dim=1)
            # print("finish forwarding augmented data")
            
            rotate_30_degrees_cluster_id = partition_func(rotate_30_degrees_data)
            rotate_60_degrees_cluster_id = partition_func(rotate_60_degrees_data)
            add_noise_cluster_id = partition_func(add_noise_data)
            change_colors_cluster_id = partition_func(change_colors_data)
            augmented_cluster_ids = torch.cat([rotate_30_degrees_cluster_id, rotate_60_degrees_cluster_id, add_noise_cluster_id, change_colors_cluster_id], dim=0)

            # pdt commented
            # rotate_30_degrees_augment = torch.cat([rotate_30_degrees_data, buffer_y.unsqueeze(1), rotate_30_degrees_logits, rotate_30_degrees_cluster_id.unsqueeze(1)], dim=1)
            # rotate_60_degrees_augment = torch.cat([rotate_60_degrees_data, buffer_y.unsqueeze(1), rotate_60_degrees_logits, rotate_60_degrees_cluster_id.unsqueeze(1)], dim=1)
            # add_noise_augment = torch.cat([add_noise_data, buffer_y.unsqueeze(1), add_noise_logits, add_noise_cluster_id.unsqueeze(1)], dim=1)
            # change_colors_augment = torch.cat([change_colors_data, buffer_y.unsqueeze(1), change_colors_logits, change_colors_cluster_id.unsqueeze(1)], dim=1)
            # pdt comment ended
            
            #augment_data = torch.cat([rotate_30_degrees_augment, rotate_60_degrees_augment, add_noise_augment, change_colors_augment], dim=0)


            buffer_cluster_ids = self.buffer.clusterID

            #Use local (?) output
            #buffer_outputs_tensor = torch.zeros((max(buffer_cluster_ids) + 1, self.buffer.buffer_size, num_classes_so_far), device=self.device)
            #augmented_outputs_tensor = torch.zeros((max(augmented_cluster_ids) + 1, self.buffer.buffer_size * 4, num_classes_so_far), device=self.device)

            #for cluster_id in range(max(buffer_cluster_ids) + 1):
                #buffer_data = self.buffer.get_data_by_clusterID(cluster_id, transform=self.transform)
                #if buffer_data is not None:
                    #buffer_x, buffer_y, buffer_logits, buffer_cluster_ids = buffer_data
                    #buffer_outputs = torch.sigmoid(self.net(buffer_x))
                    #buffer_outputs_tensor[cluster_id, :len(buffer_x), :] = buffer_outputs

                #augmented_data = augment_data[cluster_id == augmented_cluster_ids]
                #if len(augmented_data) > 0:
                # Compute output for augmented data
                    #augmented_outputs = torch.sigmoid(self.net(augmented_data))
                    #augmented_outputs_tensor[cluster_id, :len(augmented_data), :] = augmented_outputs
                    #pairwise_distances = torch.cdist(buffer_outputs_tensor[cluster_id, :len(buffer_x), :],
                                                 #augmented_outputs_tensor[cluster_id, :len(augmented_data), :])
                    #max_distance = pairwise_distances.max()
                    #loss += max_distance


            #use local (?) loss

            buffer_losses_tensor = torch.zeros(self.buffer.buffer_size, device=self.device)
            augmented_losses_tensor = torch.zeros(self.buffer.buffer_size * 4, device=self.device)
            for cluster_id in buffer_cluster_ids.long().unique():
                buffer_data = self.buffer.get_data_by_clusterID2(cluster_id, transform=self.transform, return_index=True)
                if buffer_data is not None:
                    idx, buffer_examples, buffer_labels, buffer_logits, buffer_cluster_ids = buffer_data
                    # print("forwarding buffer data")
                    # os.system("pause")
                    buffer_outputs = self.net(buffer_examples)
                    buffer_loss = F.cross_entropy(buffer_outputs, buffer_labels.long(), reduction='none')
                    buffer_losses_tensor[idx] += buffer_loss  
                    augmented_mask = (augmented_cluster_ids == cluster_id).nonzero(as_tuple=True)[0]
                    if len(augmented_mask) > 0:
                        augmented_label = augmented_labels[augmented_mask]
                        augmented_outputs = augmented_logits[augmented_mask]
                        augmented_loss = F.cross_entropy(augmented_outputs, augmented_label.long(), reduction='none')
                        augmented_losses_tensor[augmented_mask] += augmented_loss

            # max_diff = torch.zeros(buffer_cluster_ids.unique().shape[0], device=self.device)
            for cluster_id in buffer_cluster_ids.unique():
                buffer_mask = (buffer_cluster_ids == cluster_id).nonzero(as_tuple=True)[0]
                augmented_mask = (augmented_cluster_ids == cluster_id).nonzero(as_tuple=True)[0]

                if len(buffer_mask) > 0 and len(augmented_mask) > 0:
                    diff = torch.abs(buffer_losses_tensor[buffer_mask].unsqueeze(1) - augmented_losses_tensor[augmented_mask].unsqueeze(0))
                    loss += diff.max()
            # loss += max_diff.sum()

        loss.backward()

        self.opt.step()

        return loss.item(), 0, 0, 0, 0

    @staticmethod
    def binary_cross_entropy(pred, y):
        return -(pred.log() * y + (1 - y) * (1 - pred).log()).mean()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss tensor.
        :param inputs: the images to be fed to the network
        :param labels: the ground-truth labels
        :param task_idx: the task index
        :return: the differentiable loss value
        """
        labels = labels.long()
        pc = task_idx * self.dataset.N_CLASSES_PER_TASK
        ac = (task_idx + 1) * self.dataset.N_CLASSES_PER_TASK
        
        outputs, output_features = self.net(inputs, returnt='full')
        outputs = outputs[:, :ac]

        if task_idx == 0:
            # Compute loss on the current task
            targets = self.eye[labels][:, :ac]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            assert loss >= 0
        else:
            targets = self.eye[labels][:, pc:ac]
            comb_targets = torch.cat((logits[:, :pc], targets), dim=1)
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
            assert loss >= 0

        if self.args.wd_reg:
            try:
                loss += self.args.wd_reg * torch.sum(self.net.get_params() ** 2)
            except: # distributed 
                loss += self.args.wd_reg * torch.sum(self.net.module.get_params() ** 2)

        return loss, output_features

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.load_initial_checkpoint()
            self.reset_classifier()
                                
            self.net.set_return_prerelu(True)

            self.init_net(dataset)

        if self.current_task > 0:
            dataset.train_loader.dataset.targets = np.concatenate(
                [dataset.train_loader.dataset.targets,
                 self.buffer.labels.cpu().numpy()[:self.buffer.num_seen_examples]])
            if type(dataset.train_loader.dataset.data) == torch.Tensor:
                dataset.train_loader.dataset.data = torch.cat(
                    [dataset.train_loader.dataset.data, torch.stack([(
                        self.buffer.examples[i].type(torch.uint8).cpu())
                        for i in range(self.buffer.num_seen_examples)]).squeeze(1)])
            else:
                dataset.train_loader.dataset.data = np.concatenate(
                    [dataset.train_loader.dataset.data, torch.stack([((
                        self.buffer.examples[i] * 255).type(torch.uint8).cpu())
                        for i in range(self.buffer.num_seen_examples)]).numpy().swapaxes(1, 3)])


    def end_task(self, dataset) -> None:
        self.icarl_old_net = get_dataset(self.args).get_backbone().to(self.device)
        if self.args.distributed == 'dp':
            self.icarl_old_net = make_dp(self.icarl_old_net)
        _, unexpected = self.icarl_old_net.load_state_dict(deepcopy(self.net.state_dict()), strict=False)
        assert len([k for k in unexpected if 'lip_coeffs' not in k]) == 0, f"Unexpected keys in pretrained model: {unexpected}"
        self.icarl_old_net.eval()
        self.icarl_old_net.set_return_prerelu(True)

        self.net.train()
        with torch.no_grad():
            icarl_fill_buffer(self, self.buffer, dataset, self.current_task)
        self.current_task += 1
        self.class_means = None

    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels, _, _ = self.buffer.get_all_data(transform)
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)
            with bn_track_stats(self, False):
                class_means.append(self.net(x_buf, returnt='features').mean(0).flatten())
        self.class_means = torch.stack(class_means)
