import os
import torch
from models import Autoformer, DLinear, FEDformer, PatchTST, MICN, Crossformer, iTransformer, MambaTS, FourierGNN


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'PatchTST': PatchTST,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'iTransformer': iTransformer,
            'FourierGNN': FourierGNN,
            'MambaTS': MambaTS,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        # from torchsummary import summary
        # from torchinfo import summary
        # # torch.Size([16, 720, 321]) torch.Size([16, 768, 321]) torch.Size([16, 720, 4]) torch.Size([16, 768, 4])
        # # print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
        # summary(self.model, [(self.args.batch_size, self.args.seq_len, self.args.enc_in),
        #                      (self.args.batch_size, self.args.seq_len, self.args.enc_in),
        #                      (self.args.batch_size, self.args.seq_len, 4),
        #                      (self.args.batch_size, self.args.seq_len, 4)], device='cuda')

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
