import argparse

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TokenLabelConverter(object):
    """ Convert between text-label and text-index """
    def __init__(self, opt):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.SPACE = '[s]'
        self.GO = '[GO]'
        
        self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + list(opt.character)
        
        self.dict = { word: i for i, word in enumerate(self.character) }
        self.batch_max_length = opt.batch_max_length + len(self.list_token)
    
    def encode(self, text):
        """ convert text-label into text-index.
        """
        _ = [len(s) + len(self.list_token) for s in text]  # +2 for [GO] and [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            txt = [self.GO] + list(t) + [self.SPACE]
            txt = [self.dict[char] for char in txt]
            batch_text[i][:len(txt)] = torch.LongTensor(txt)  # batch_text[:, 0] = [GO] token
        return batch_text.to(device)
    
    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, _ in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""
    def __init__(self):
        self.reset()
    
    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v
    
    def reset(self):
        self.n_count = 0
        self.sum = 0
    
    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def get_device(verbose=True):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if verbose:
        print('Device:', device)
    return device


def get_args(is_train=True):
    parser = argparse.ArgumentParser(description='STR')
    
    # for test
    parser.add_argument('--eval_data', required=not is_train, help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--calculate_infer_time', action='store_true', help='calculate inference timing')
    parser.add_argument('--flops', action='store_true', help='calculates approx flops (may not work)')
    
    # for train
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=is_train, help='path to training dataset')
    parser.add_argument('--valid_data', required=is_train, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument(
      '--workers', type=int, help='number of data loading workers. Use -1 to use all cores.', default=4
    )
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help='path to model to continue training')
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--sgd', action='store_true', help='Whether to use SGD (default is Adadelta)')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument(
      '--select_data',
      type=str,
      default='/',
      help='select training data (default is MJ-ST, which means MJ and ST used as training data)'
    )
    parser.add_argument(
      '--batch_ratio', type=str, default='1.0', help='assign ratio for each selected data in the batch'
    )
    parser.add_argument(
      '--total_data_usage_ratio',
      type=str,
      default='1.0',
      help='total data usage ratio, this ratio is multiplied to total number of data.'
    )
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    choices = [
      'vit_t_16',
      'vit_s_16',
      'vit_b_16',
    ]
    parser.add_argument('--transformer', default=choices[0], help='Which vit/deit transformer model', choices=choices)
    parser.add_argument('--transformation', type=str, default=None, help='Transformation stage. None|TPS')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument(
      '--output_channel', type=int, default=512, help='the number of output channel of Feature extractor'
    )
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    
    # selective augmentation
    # can choose specific data augmentation
    parser.add_argument('--issel_aug', action='store_true', help='Select augs')
    parser.add_argument('--sel_prob', type=float, default=1., help='Probability of applying augmentation')
    parser.add_argument('--pattern', action='store_true', help='Pattern group')
    parser.add_argument('--warp', action='store_true', help='Warp group')
    parser.add_argument('--geometry', action='store_true', help='Geometry group')
    parser.add_argument('--weather', action='store_true', help='Weather group')
    parser.add_argument('--noise', action='store_true', help='Noise group')
    parser.add_argument('--blur', action='store_true', help='Blur group')
    parser.add_argument('--camera', action='store_true', help='Camera group')
    parser.add_argument('--process', action='store_true', help='Image processing routines')
    
    # use cosine learning rate decay
    parser.add_argument('--scheduler', action='store_true', help='Use lr scheduler')
    
    parser.add_argument('--intact_prob', type=float, default=0.5, help='Probability of not applying augmentation')
    parser.add_argument('--isrand_aug', action='store_true', help='Use RandAug')
    parser.add_argument('--augs_num', type=int, default=3, help='Number of data augment groups to apply. 1 to 8.')
    parser.add_argument(
      '--augs_mag', type=int, default=None, help='Magnitude of data augment groups to apply. None if random.'
    )
    
    # for comparison to other augmentations
    parser.add_argument('--issemantic_aug', action='store_true', help='Use Semantic')
    parser.add_argument('--isrotation_aug', action='store_true', help='Use ')
    parser.add_argument('--isscatter_aug', action='store_true', help='Use ')
    parser.add_argument('--islearning_aug', action='store_true', help='Use ')
    
    # orig paper uses this for fast benchmarking
    parser.add_argument('--fast_acc', action='store_true', help='Fast average accuracy computation')
    
    parser.add_argument('--infer_model', type=str, default=None, help='generate inference jit model')
    parser.add_argument('--quantized', action='store_true', help='Model quantization')
    parser.add_argument('--static', action='store_true', help='Static model quantization')
    args = parser.parse_args()
    return args
