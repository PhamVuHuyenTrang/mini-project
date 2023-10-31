import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import List
from models.utils.continual_model import ContinualModel

def add_regularization_args(parser):
    # Lambda parameter for robustness with respect to loss fucntion
    parser.add_argument('--loss_reg', type=float, required=False, default=0,
                        help='Lambda parameter for robustness with respect to loss fucntion')

    # Lambda parameter for for robustness with respect to linear function
    parser.add_argument('--linear_reg', type=float, required=False, default=0,
                        help='Lambda parameter for for robustness with respect to linear function')


class RobustnessOptimizer(ContinualModel):

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone,loss,args,transform)
        self.args = args


    def to(self, device):
        self.device = device
        return super().to(device)
  

    @torch.no_grad()
    def init_net(self, dataset):
        # Eval L for initial model
        self.net.eval()
        
        all_lips = []
        for i, (inputs, labels, _) in enumerate(tqdm(dataset.train_loader, desc="Evaluating initial L")):
            if i>3 and self.args.debug_mode:
                continue
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if len(inputs.shape) == 5:
                B, n, C, H, W = inputs.shape
                inputs = inputs.view(B*n, C, H, W)
            else:
                B, C, H, W = inputs.shape

            _, partial_features = self.net(inputs, returnt='full')

            #lip_inputs = [inputs] + partial_features[:-1]

            #lip_values = self.get_feature_lip_coeffs(lip_inputs)
            # (B, F)
            #lip_values = torch.stack(lip_values, dim=1)

            #all_lips.append(lip_values)
            
        #self.budget_lip = torch.cat(all_lips, dim=0).mean(0)
        
        inp = next(iter(dataset.train_loader))[0]
        _, teacher_feats = self.net(inp.to(self.device), returnt='full')

        #self.net.lip_coeffs = torch.autograd.Variable(torch.randn(len(teacher_feats)-1, dtype=torch.float), requires_grad=True).to(self.device)
        #self.net.lip_coeffs.data = self.budget_lip.detach().clone()
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr,
                        weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)

        self.net.train()


    def buffer_lip_loss(self, features: List[torch.Tensor]) -> torch.Tensor:
        lip_values = self.get_feature_lip_coeffs(features)
        # (B, F)
        lip_values = torch.stack(lip_values, dim=1)

        return lip_values.mean()

    def budget_lip_loss(self, features: List[torch.Tensor]) -> torch.Tensor:
        loss = 0
        lip_values = self.get_feature_lip_coeffs(features)
        # (B, F)
        lip_values = torch.stack(lip_values, dim=1)

        if self.args.headless_init_act == "relu":
            tgt = F.relu(self.net.lip_coeffs[:len(lip_values[0])])
        elif self.args.headless_init_act == "lrelu":
            tgt = F.leaky_relu(self.net.lip_coeffs[:len(lip_values[0])])
        else:
            assert False
        tgt = tgt.unsqueeze(0).expand(lip_values.shape)

        loss += F.l1_loss(lip_values, tgt)

        return loss

    def get_norm(self, t: torch.Tensor):
        return torch.norm(t, dim=1, keepdim=True)+torch.finfo(torch.float32).eps
    
    def measure_lip_base(self, s_feats_a, s_feats_b, t_feats_a, t_feats_b):
        with torch.no_grad():
            s_feats_a, s_feats_b = s_feats_a / self.get_norm(s_feats_a), s_feats_b / self.get_norm(s_feats_b)
            t_feats_a, t_feats_b = t_feats_a / self.get_norm(t_feats_a), t_feats_b / self.get_norm(t_feats_b)

            TM_s = self.compute_transition_matrix(s_feats_a, s_feats_b)
            TM_t = self.compute_transition_matrix(t_feats_a, t_feats_b)

            L_s = self.top_eigenvalue(K=TM_s).mean().item()
            L_t = self.top_eigenvalue(K=TM_t).mean().item()

        return L_s, L_t
