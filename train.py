import os
import random
import re
import string
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset_old import AlignCollate, Batch_Balanced_Dataset, hierarchical_dataset
from test import get_infer_model, validation
from utils import Average, get_args, TokenLabelConverter
from vit import create_vit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    opt.eval = False
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    opt.eval = True
    if opt.sensitive:
        opt.data_filtering_off = True
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid,
        pin_memory=True
    )
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    """ model configuration """
    if opt.transformer:
        converter = TokenLabelConverter(opt)
    else:
        converter = None
    opt.num_classes = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = create_vit(opt.transformer, opt.num_classes)

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    """ setup loss """
    # README: https://github.com/clovaai/deep-text-recognition-benchmark/pull/209
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss average
    loss_avg = Average()

    # filter that only requires gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    # setup optimizer
    scheduler = None
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)

    if opt.scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_iter)

    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        opt_file.write(opt_log)
        total_params = int(sum(params_num))
        total_params = f'Trainable network params num : {total_params:,}'
        print(total_params)
        opt_file.write(total_params)
    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except Exception:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    while True:
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)

        target = converter.encode(labels)
        preds = model(image, seqlen=converter.batch_max_length)
        cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0:
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, _, _ = validation(
                        model, criterion, valid_loader, converter, opt
                    )
                model.train()

                # training loss and validation loss
                loss_log = f'[{iteration + 1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, ' \
                           f'Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, ' \
                                    f'{"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                    get_infer_model(model, opt, save_name='best_acc.pt')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                    get_infer_model(model, opt, save_name='best_norm.pt')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, ' \
                                 f'{"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    pred = pred[:pred.find('[s]')]

                    # To evaluate 'case-sensitive model' with alphanumeric and case-insensitive setting.
                    if opt.sensitive and opt.data_filtering_off:
                        pred = pred.lower()
                        gt = gt.lower()
                        alphanumeric_case_insensitive = '0123456789abcdefghijklmnopqrstuvwxyz'
                        out_of_alphanumeric_case_insensitive = f'[^{alphanumeric_case_insensitive}]'
                        pred = re.sub(out_of_alphanumeric_case_insensitive, '', pred)
                        gt = re.sub(out_of_alphanumeric_case_insensitive, '', gt)

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e5 iter.
        if (iteration + 1) % 1e3 == 0:
            torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration + 1}.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1
        if scheduler is not None:
            scheduler.step()


if __name__ == '__main__':
    opt = get_args()
    if not opt.exp_name:
        opt.exp_name = f'{opt.transformer}'

    opt.exp_name += f'-Seed{opt.manualSeed}'

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)
    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    """ Seed and GPU setting """
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    if opt.workers <= 0:
        opt.workers = (int(os.cpu_count()) // 2) // opt.num_gpu

    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

    train(opt)
