import os
import re
import string
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import validators
from nltk.metrics.distance import edit_distance

from dataset_old import AlignCollate, hierarchical_dataset
from utils import Average, get_args, TokenLabelConverter
from vit import create_vit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def benchmark_all_eval(model, criterion, converter, opt):
    """ evaluation with 10 benchmark evaluation datasets """

    if opt.fast_acc:
        # To easily compute the total accuracy of our paper.
        eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']
    else:
        # The evaluation datasets, dataset order is the same with Table 1 in our paper.
        eval_data_list = [
            'IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857', 'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP',
            'CUTE80'
        ]

    if opt.calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data,
            batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation,
            pin_memory=True
        )

        _, accuracy_by_best_model, norm_ED_by_best_model, _, _, _, infer_time, length_of_data = validation(
            model, criterion, evaluation_loader, converter, opt
        )
        list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        log.write(eval_data_log)
        print(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}')
        log.write(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}\n')
        print(dashed_line)
        log.write(dashed_line + '\n')

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num / 1e6:0.3f}'
    if opt.flops:
        evaluation_log += get_flops(model, opt, converter)
    print(evaluation_log)
    log.write(evaluation_log + '\n')
    log.close()

    return None


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Average()

    for _, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        target = converter.encode(labels)

        start_time = time.time()
        preds = model(image, seqlen=converter.batch_max_length)
        _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, converter.batch_max_length)
        forward_time = time.time() - start_time
        cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

        length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
        preds_str = converter.decode(preds_index[:, 1:], length_for_pred)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after 'end of sentence' token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case-sensitive model' with alphanumeric and case-insensitive setting.
            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitive = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitive = f'[^{alphanumeric_case_insensitive}]'
                pred = re.sub(out_of_alphanumeric_case_insensitive, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitive, '', gt)

            if pred == gt:
                n_correct += 1

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except Exception:
                confidence_score = 0  # for empty pred case, when prune after 'end of sentence' token ([s])
            confidence_score_list.append(confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data


def get_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    return new_state_dict


# https://pytorch.org/tutorials/beginner/saving_loading_models.html
def get_infer_model(model, opt, save_name=''):
    new_state_dict = get_state_dict(model.state_dict())
    model = create_vit(opt.transformer, opt.num_classes)
    model.load_state_dict(new_state_dict)
    model.eval()
    if opt.quantized:
        # static quantization: Work in progress
        if opt.static:
            backend = 'qnnpack'
            model.qconfig = torch.quantization.get_default_qconfig(backend)
            torch.backends.quantized.engine = backend
            model_quantized = torch.quantization.prepare(model, inplace=False)
            model_quantized = torch.quantization.convert(model_quantized, inplace=False)
        # support for dynamic quantization
        else:
            from torch.quantization import quantize_dynamic
            model_quantized = quantize_dynamic(
                model=model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8, inplace=False
            )
        # quantized model save/load https://pytorch.org/docs/stable/quantization.html
        model = torch.jit.script(model_quantized)

    model_scripted = torch.jit.script(model)
    model_scripted.save(save_name)


def test(opt):
    """ model configuration """
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = create_vit(opt.transformer, opt.num_classes)

    print(
        'model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
        opt.hidden_size, opt.num_class, opt.batch_max_length, opt.transformation
    )
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)

    if validators.url(opt.saved_model):
        model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    else:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    if opt.infer_model is not None:
        get_infer_model(model, opt)
        return
    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')
    """ setup loss """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    """ evaluation """
    model.eval()
    opt.eval = True
    with torch.no_grad():
        if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
            benchmark_all_eval(model, criterion, converter, opt)
        else:
            log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a')
            AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
            eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation,
                pin_memory=True
            )
            _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
                model, criterion, evaluation_loader, converter, opt
            )
            log.write(eval_data_log)
            print(f'{accuracy_by_best_model:0.3f}')
            log.write(f'{accuracy_by_best_model:0.3f}\n')
            log.close()


# https://github.com/clovaai/deep-text-recognition-benchmark/issues/125
def get_flops(model, opt, converter):
    from thop import profile
    inputs = torch.randn(1, 1, opt.imgH, opt.imgW).to(device)
    model = model.to(device)
    seqlen = converter.batch_max_length
    text_for_pred = torch.LongTensor(1, seqlen).fill_(0).to(device)
    MACs, _ = profile(model, inputs=(inputs, text_for_pred, True, seqlen))

    flops = 2 * MACs  # approximate FLOPS
    return f'Approximate FLOPS: {flops:0.3f}'


if __name__ == '__main__':
    opt = get_args(is_train=False)
    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)
