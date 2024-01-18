from copy import deepcopy

from functions.augmentations import normalize
import torch
import torch.nn as nn
from torchvision.transforms import RandomHorizontalFlip, RandomResizedCrop, ColorJitter, RandomGrayscale

import torch.nn.functional as F
from torchvision import transforms
from datasets import get_dataset
from functions.buffer import Buffer
from functions.args import *
from models.utils.continual_model import ContinualModel
from functions.distributed import make_dp
from functions.lipschitz import RobustnessOptimizer, add_regularization_args
from functions.create_partition import create_partition_func_1nn
from functions.no_bn import bn_track_stats
import numpy as np
from functions.augmentations import (
    rotate_30_degrees,
    rotate_60_degrees,
    add_noise,
    change_colors,
)

partition_func = create_partition_func_1nn((84, 84, 3), n_centroids=5000)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Continual Learning via iCaRL."
        "Treated with Lipschitz constraints!"
    )

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)
    add_regularization_args(parser)

    parser.add_argument(
        "--wd_reg",
        type=float,
        required=True,
        help="L2 regularization applied to the parameters.",
    )

    return parser


def icarl_fill_buffer(
    self: ContinualModel, mem_buffer: Buffer, dataset, t_idx: int
) -> None:
    """
    Adds examples from the current task to the memory buffer
    by means of the herding strategy.
    :param mem_buffer: the memory buffer
    :param dataset: the dataset from which take the examples
    :param t_idx: the task index
    """

    mode = self.net.training
    self.net.eval()
    samples_per_class = mem_buffer.buffer_size // (
        dataset.N_CLASSES_PER_TASK * (t_idx + 1)
    )

    if t_idx > 0:
        # 1) First, subsample prior classes
        buf_x, buf_y, buf_l, buf_clusterID = self.buffer.get_all_data()

        mem_buffer.empty()
        for _y in buf_y.unique():
            idx = buf_y == _y
            _y_x, _y_y, _y_l, _y_clusterID = (
                buf_x[idx],
                buf_y[idx],
                buf_l[idx],
                buf_clusterID[idx],
            )
            mem_buffer.add_data(
                examples=_y_x[:samples_per_class],
                labels=_y_y[:samples_per_class],
                logits=_y_l[:samples_per_class],
                clusterID=_y_clusterID[:samples_per_class],
            )

    # 2) Then, fill with current tasks
    loader = dataset.train_loader
    mean, std = (
        dataset.get_denormalization_transform().mean,
        dataset.get_denormalization_transform().std,
    )
    classes_start, classes_end = (
        t_idx * dataset.N_CLASSES_PER_TASK,
        (t_idx + 1) * dataset.N_CLASSES_PER_TASK,
    )
    # # todo add normalize to features for other datasets

    # 2.1 Extract all features
    a_x, a_y, a_f, a_l = [], [], [], []
    for x, y, not_norm_x in loader:
        mask = (y >= classes_start) & (y < classes_end)
        x, y, not_norm_x = x[mask], y[mask], not_norm_x[mask]
        if not x.size(0):
            continue
        x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
        a_x.append(not_norm_x.to("cpu"))
        a_y.append(y.to("cpu"))
        try:
            feats = self.net.features(normalize(not_norm_x, mean, std)).float()
            outs = self.net.classifier(feats)
        except:
            replica_hack = False
            if (
                self.args.distributed == "dp"
                and len(not_norm_x) < torch.cuda.device_count()
            ):
                # yup, that's exactly right! I you have more GPUs than inputs AND you use kwargs,
                # dataparallel breaks down. So we pad with mock data and then ignore the padding.
                # ref https://github.com/pytorch/pytorch/issues/31460
                replica_hack = True
                not_norm_x = not_norm_x.repeat(torch.cuda.device_count(), 1, 1, 1)

            outs, feats = self.net(normalize(not_norm_x, mean, std), returnt="both")

            if replica_hack:
                outs, feats = (
                    outs.split(len(not_norm_x) // torch.cuda.device_count())[0],
                    feats.split(len(not_norm_x) // torch.cuda.device_count())[0],
                )

        a_f.append(feats.cpu())
        a_l.append(torch.sigmoid(outs).cpu())
    a_x, a_y, a_f, a_l = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f), torch.cat(a_l)

    # 2.2 Compute class means
    for _y in range(classes_start, classes_end):
        idx = a_y == _y
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
                examples=_x[idx_min : idx_min + 1].to(self.device),
                labels=_y[idx_min : idx_min + 1].to(self.device),
                logits=_l[idx_min : idx_min + 1].to(self.device),
                clusterID=partition_func(_x[idx_min : idx_min + 1]).to(self.device),
            )

            running_sum += feats[idx_min : idx_min + 1]
            feats[idx_min] = feats[idx_min] + 1e6
            i += 1

    assert len(mem_buffer.examples) <= mem_buffer.buffer_size
    assert mem_buffer.num_seen_examples <= mem_buffer.buffer_size
    self.net.train(mode)


class ICarlLipschitz(RobustnessOptimizer):
    NAME = "icarl_lipschitz"
    COMPATIBILITY = ["class-il", "task-il"]

    def __init__(self, backbone, loss, args, transform):
        super(ICarlLipschitz, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS).to(
            self.device
        )

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
            feats = self.net(x, returnt="both")[1].float().squeeze()

        feats = feats.reshape(feats.shape[0], -1)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def observe(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        not_aug_inputs: torch.Tensor,
        logits=None,
        epoch=None,
    ):
        if not hasattr(self, "classes_so_far_buffer"):
            self.classes_so_far_buffer = labels.unique().to(labels.device)
            self.register_buffer("classes_so_far", self.classes_so_far_buffer)
        else:
            self.classes_so_far_buffer = torch.cat(
                (
                    self.classes_so_far_buffer,
                    labels.to(self.classes_so_far_buffer.device),
                )
            ).unique()

            self.register_buffer("classes_so_far", self.classes_so_far_buffer)

        self.class_means = None
        if self.current_task > 0:
            with torch.no_grad():
                logits = torch.sigmoid(self.icarl_old_net(inputs))
        self.opt.zero_grad()

        transform = nn.Sequential(
                RandomResizedCrop(size=(84, 84), scale=(0.2, 1.)),
                RandomHorizontalFlip(),
                ColorJitter(0.4, 0.4, 0.4, 0.1),
                RandomGrayscale(p=0.2)
            )
        augment = transform(inputs)
        inputs = torch.cat([inputs, augment], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        if logits is not None:
            logits = torch.cat([logits, logits], dim=0)

        loss, _ = self.get_loss(inputs, labels, self.current_task, logits)

        loss.backward()

        self.opt.step()
        torch.cuda.empty_cache()
        return loss.item(), 0, 0, 0, 0

    @staticmethod
    def binary_cross_entropy(pred, y):
        return -(pred.log() * y + (1 - y) * (1 - pred).log()).mean()

    def get_loss(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        task_idx: int,
        logits: torch.Tensor,
    ) -> torch.Tensor:
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

        outputs, output_features = self.net(inputs, returnt="full")
        #print("input___", inputs)
        #print("outputs", outputs)
        outputs = outputs[:, :ac]

        if task_idx == 0:
            # Compute loss on the current task
            targets = self.eye[labels][:, :ac]
            loss_ce = F.binary_cross_entropy_with_logits(outputs, targets)
            assert loss_ce >= 0
        else:
            targets = self.eye[labels][:, pc:ac]
            comb_targets = torch.cat((logits[:, :pc], targets), dim=1)
            #print("logits", logits[:, :pc])
            #print("targets", targets)
            loss_ce = F.binary_cross_entropy_with_logits(outputs, comb_targets)
            #print("comb_targets" ,comb_targets)
            #print("outputs", outputs)
            #print("loss_ce", loss_ce)
            assert loss_ce >= 0

        if self.args.wd_reg:
            try:
                loss_wd = self.args.wd_reg * torch.sum(self.net.get_params() ** 2)
            except:  # distributed
                loss_wd = self.args.wd_reg * torch.sum(
                    self.net.module.get_params() ** 2
                )
        else:
            loss_wd = 0

        # Robustness losses (New regularization)
        unique_labels = torch.unique(labels)
        num_classes_so_far = unique_labels.numel()

        loss_reg = torch.zeros_like(loss_ce).to(self.device)

        if not self.buffer.is_empty():
            if self.args.method == 'lider':
                lip_inputs = [inputs] + output_features[:-1]
                
                loss_reg = self.args.buffer_lip_lambda * self.buffer_lip_loss(lip_inputs) + self.args.budget_lip_lambda * self.budget_lip_loss(lip_inputs)

            elif self.args.method == 'localrobustness':
                (
                    choice,
                    buffer_x,
                    buffer_y,
                    buffer_logits,
                    buffer_cluster_ids,
                ) = self.buffer.get_data(
                    self.setting.minibatch_size, transform=self.transform, return_index=True
                )

                (
                    augment_examples,
                    augmented_labels,
                    _,
                    augmented_cluster_ids,
                ) = self.buffer.get_augment_data(choice)
                

                augment_output, augment_features = self.net(
                    augment_examples, returnt="full"
                )
                #print("augment_examples", augment_examples)
                #print("augment_output", augment_output)
                buffer_output, buffer_feature = self.net(buffer_x, returnt="full")
                #print("buffer_output", buffer_output)
                reg = 0.01
                mean = 1/(len(augment_features) * 4 * (self.buffer.buffer_size ** 2))
                for af, bf in zip(augment_features, buffer_feature):
                    #print("bf.shape[0]", bf.shape[0])
                    bf = torch.cat([bf] * (af.shape[0] // bf.shape[0]))
                    if len(bf.shape) == 2:
                        distance = torch.sqrt(((bf - af) ** 2).sum(dim = (1,)))
                    else:
                        distance = torch.sqrt(((bf - af) ** 2).sum(dim=(1, 2, 3)))
                    loss_reg += reg * mean * distance.sum()
                    #print("loss_reg", loss_reg)

        print(f'loss ce: {loss_ce}, loss wd: {loss_wd}, loss_reg: {loss_reg}')
        loss = loss_ce + loss_wd + loss_reg
        return loss, output_features

    def begin_task(self, dataset):
        if self.current_task == 0:
            torch.use_deterministic_algorithms(True)
            self.load_initial_checkpoint()
            self.reset_classifier()

            self.net.set_return_prerelu(True)

            self.init_net(dataset)

        if self.current_task > 0:
            torch.use_deterministic_algorithms(False)
            dataset.train_loader.dataset.targets = np.concatenate(
                [
                    dataset.train_loader.dataset.targets,
                    self.buffer.labels.cpu().numpy()[: self.buffer.num_seen_examples],
                ]
            )
            if type(dataset.train_loader.dataset.data) == torch.Tensor:
                dataset.train_loader.dataset.data = torch.cat(
                    [
                        dataset.train_loader.dataset.data,
                        torch.stack(
                            [
                                (self.buffer.examples[i].type(torch.uint8).cpu())
                                for i in range(self.buffer.num_seen_examples)
                            ]
                        ).squeeze(1),
                    ]
                )
            else:
                dataset.train_loader.dataset.data = np.concatenate(
                    [
                        dataset.train_loader.dataset.data,
                        torch.stack(
                            [
                                (
                                    (self.buffer.examples[i] * 255)
                                    .type(torch.uint8)
                                    .cpu()
                                )
                                for i in range(self.buffer.num_seen_examples)
                            ]
                        )
                        .numpy()
                        .swapaxes(1, 3),
                    ]
                )

    def end_task(self, dataset) -> None:
        self.icarl_old_net = get_dataset(self.args).get_backbone().to(self.device)
        if self.args.distributed == "dp":
            self.icarl_old_net = make_dp(self.icarl_old_net)
        _, unexpected = self.icarl_old_net.load_state_dict(
            deepcopy(self.net.state_dict()), strict=False
        )
        assert (
            len([k for k in unexpected if "lip_coeffs" not in k]) == 0
        ), f"Unexpected keys in pretrained model: {unexpected}"
        self.icarl_old_net.eval()
        self.icarl_old_net.set_return_prerelu(True)

        self.net.train()
        with torch.no_grad():
            icarl_fill_buffer(self, self.buffer, dataset, self.current_task)
            mean, std = (
                self.dataset.get_denormalization_transform().mean,
                self.dataset.get_denormalization_transform().std,
            )
            self.buffer.generate_augment_data(mean, std, partition_func)
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
            x_buf_list = [
                examples[i] for i in range(0, len(examples)) if labels[i].cpu() == _y
            ]

            if x_buf_list:
                x_buf = torch.stack(x_buf_list, dim=0).to(self.device)
                with bn_track_stats(self, False):
                    class_means.append(
                        self.net(x_buf, returnt="features").mean(0).flatten()
                    )
            else:
                x_buf_list_dummy = [examples[0]]
                x_buf_dummy = torch.stack(x_buf_list_dummy, dim=0).to(self.device)

                class_means.append(
                    torch.zeros_like(
                        self.net(x_buf_dummy, returnt="features").mean(0).flatten()
                    )
                )
        self.class_means = torch.stack(class_means)
