import os
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch._utils import _accumulate
from torch.utils.data import ConcatDataset, Subset


class Batch_Balanced_Dataset(object):
    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is 'MJ-ST' and batch_ratio is '0.5-0.5',
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(
            f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n'
        )
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)
            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [
                Subset(_dataset, indices[offset - length:offset])
                for offset, length in zip(_accumulate(dataset_split), dataset_split)
            ]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x ' \
                             f'{opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x ' \
                              f'{float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset,
                batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate,
                pin_memory=True
            )
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = next(data_loader_iter)
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data: Union[str, List] = '/'):
    """ select_data='/' contains all subdirectories of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, _ in os.walk(root + '/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
