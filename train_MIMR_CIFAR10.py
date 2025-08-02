import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import argparse
import copy
import torch.nn.functional as F
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from cutout import Cutout
from CIFAR10_Models import *
from utils_1 import *
from Feature_model.feature_resnet import *
from torchvision import datasets, transforms
import logging

logger = logging.getLogger(__name__)

def parse_arguments():
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument('--data-dir', default='CIFAR10', type=str)
    config_parser.add_argument('--batch-size', default=128, type=int)
    config_parser.add_argument('--epochs', default=110, type=int)
    config_parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    config_parser.add_argument('--weight-decay', default=5e-4, type=float)
    config_parser.add_argument('--momentum', default=0.9, type=float)
    config_parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    config_parser.add_argument('--epsilon', default=8, type=int)
    config_parser.add_argument('--lr-min', default=0., type=float)
    config_parser.add_argument('--lr-max', default=0.1, type=float)
    config_parser.add_argument('--c_num', default=0.125, type=float)
    config_parser.add_argument('--EMA_value', default=0.82, type=float)
    config_parser.add_argument('--alpha', default=8, type=float, help='Step size')
    config_parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    config_parser.add_argument('--out_dir', default='output', type=str, help='Output directory')
    config_parser.add_argument('--seed', default=0, type=int, help='Random seed')
    config_parser.add_argument('--normal_mean', default=0, type=float, help='normal_mean')
    config_parser.add_argument('--normal_std', default=1, type=float, help='normal_std')
    config_parser.add_argument('--lamda', default=12, type=float, help='Label Smoothing')
    config_parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    config_parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
    config_parser.add_argument('--factor', default=0.7, type=float)
    config_parser.add_argument('--length', type=int, default=4, help='length of the holes')
    config_parser.add_argument('--max-norm', default=1.0, type=float, help='Maximum norm value for regularization')
    return config_parser.parse_args()

config = parse_arguments()

def get_loaders_cutout(dir_, batch_size):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    train_transforms.transforms.append(Cutout(n_holes=config.n_holes, length=config.length))

    num_workers = 0
    train_data = datasets.CIFAR10(
        dir_, train=True, transform=train_transforms, download=True)
    test_data = datasets.CIFAR10(
        dir_, train=False, transform=test_transforms, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
       batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True 
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader

import numpy as np
from torch.autograd import Variable

def compute_smoothed_labels(target_labels, smoothing_factor):
    one_hot_encoded = np.eye(10)[target_labels.cuda().data.cpu().numpy()]
    smoothed_labels = one_hot_encoded * smoothing_factor + (one_hot_encoded - 1.) * ((smoothing_factor - 1) / float(10 - 1))
    return smoothed_labels

def compute_smoothed_loss(model_input, target):
    log_probs = F.log_softmax(model_input, dim=-1)
    loss_value = (-target * log_probs).sum(dim=-1).mean()
    return loss_value

class ExponentialMovingAverage(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
        self.step_counter = 0
        self.model = copy.deepcopy(model)
        self.decay_factor = alpha
        self.use_buffer_ema = buffer_ema
        self.shadow_state = self.get_model_state()
        self.backup_state = {}
        self.param_names = [k for k, _ in self.model.named_parameters()]
        self.buffer_names = [k for k, _ in self.model.named_buffers()]

    def update_parameters(self, model):
        current_decay = min(self.decay_factor, (self.step_counter + 1) / (self.step_counter + 10))
        model_state = model.state_dict()
        for name in self.param_names:
            self.shadow_state[name].copy_(
                current_decay * self.shadow_state[name] + (1 - current_decay) * model_state[name])
        for name in self.buffer_names:
            if self.use_buffer_ema:
                self.shadow_state[name].copy_(
                    current_decay * self.shadow_state[name] + (1 - current_decay) * model_state[name])
            else:
                self.shadow_state[name].copy_(model_state[name])
        self.step_counter += 1

    def apply_shadow_weights(self):
        self.backup_state = self.get_model_state()
        self.model.load_state_dict(self.shadow_state)

    def restore_original_weights(self):
        self.model.load_state_dict(self.backup_state)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

def execute_training():
    config = parse_arguments()
    output_directory = os.path.join(config.out_dir, 'Ours')
    output_directory = os.path.join(output_directory, 'epsilon_' + str(config.epsilon))
    output_directory = os.path.join(output_directory, 'alpha_' + str(config.alpha))
    output_directory = os.path.join(output_directory, 'model_' + str(config.model))
    output_directory = os.path.join(output_directory, 'factor_' + str(config.factor))
    output_directory = os.path.join(output_directory, 'length_' + str(config.length))
    output_directory = os.path.join(output_directory, 'EMA_value_' + str(config.EMA_value))
    output_directory = os.path.join(output_directory, 'lamda_' + str(config.lamda))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    log_path = os.path.join(output_directory, 'output.log')
    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(output_directory, 'output.log'))
    logger.info(config)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    train_loader, test_loader = get_loaders_cutout(config.data_dir, config.batch_size)

    eps_value = (config.epsilon / 255.) / std
    alpha_value = (config.alpha / 255.) / std

    print('==> Building model..')
    logger.info('==> Building model..')
    if config.model == "VGG":
        model = VGG('VGG19')
    elif config.model == "ResNet18":
        model = Feature_ResNet18()
    elif config.model == "PreActResNest18":
        model = PreActResNet18()
    elif config.model == "WideResNet":
        model = WideResNet()
    model = model.cuda()
    model.train()
    teacher_model = ExponentialMovingAverage(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr_max, momentum=config.momentum, weight_decay=config.weight_decay)

    perturbation = torch.zeros(config.batch_size, 3, 32, 32).cuda()
    if config.delta_init == 'previous':
         perturbation = torch.zeros_like(perturbation).cuda()

    # Initialize momentum buffer for delta
    grad_momentum = torch.zeros_like(perturbation).cuda()

    total_steps = config.epochs * len(train_loader)
    if config.lr_schedule == 'cyclic':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.lr_min, max_lr=config.lr_max,
                                                      step_size_up=total_steps / 2, step_size_down=total_steps / 2)
    elif config.lr_schedule == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[total_steps * 99 / 110, total_steps * 104 / 110],
                                                         gamma=0.1)

    epoch_train_clean_list = []
    epoch_train_pgd_list = []
    epoch_clean_list = []
    epoch_pgd_list = []
    initial_loss = []
    initial_accuracy = []
    training_loss = []
    training_accuracy = []
    best_accuracy = 0
    for epoch_index in range(config.epochs):
        epoch_duration = 0
        total_loss = 0
        total_accuracy = 0
        initial_batch_loss = 0
        initial_batch_accuracy = 0
        samples_processed = 0
        teacher_model.model.eval()
        for batch_index, (input_data, target_labels) in enumerate(train_loader):
            batch_start = time.time()
            input_data, target_labels = input_data.cuda(), target_labels.cuda()
            if batch_index == 0:
                first_batch_data = (input_data, target_labels)
            if config.delta_init != 'previous':
                perturbation = torch.zeros_like(input_data).cuda()
            if config.delta_init == 'random':
                for j in range(len(eps_value)):
                    perturbation[:, j, :, :].uniform_(-eps_value[j][0][0].item(), eps_value[j][0][0].item())
                perturbation.data = clamp(perturbation, lower_limit - input_data, upper_limit - input_data)

            perturbation = eps_value / 2 * torch.sign(perturbation)
            perturbation.data = clamp(perturbation, lower_limit - input_data, upper_limit - input_data)
            perturbation.requires_grad = True

            adv_output, original_features = model(input_data + perturbation[:input_data.size(0)])
            perturbation_clone = perturbation.clone().detach()
            adv_output = torch.nn.Softmax(dim=1)(adv_output)
            original_features = torch.nn.Softmax(dim=1)(original_features)

            loss_value = F.cross_entropy(adv_output, target_labels)

            adv_output, original_features = model(input_data)

            adv_output = torch.nn.Softmax(dim=1)(adv_output)
            original_features = torch.nn.Softmax(dim=1)(original_features)

            initial_batch_loss += loss_value.item() * target_labels.size(0)
            initial_batch_accuracy += (adv_output.max(1)[1] == target_labels).sum().item()

            loss_value.backward(retain_graph=True)
            perturbation_gradient = perturbation.grad.detach()
            perturbation.data = clamp(perturbation + alpha_value * torch.sign(perturbation_gradient), -eps_value, eps_value)
            perturbation.data[:input_data.size(0)] = clamp(perturbation[:input_data.size(0)], lower_limit - input_data, upper_limit - input_data)

            # Update delta using momentum
            if grad_momentum.size(0) != perturbation_gradient.size(0):
              grad_momentum = grad_momentum[:perturbation_gradient.size(0)] 
            grad_momentum = config.momentum * grad_momentum + (1 - config.momentum) * perturbation_gradient
            perturbation = perturbation + alpha_value * grad_momentum

            perturbation.data = clamp(perturbation, lower_limit - input_data, upper_limit - input_data)
            perturbation = perturbation.detach()

            model_output, feature_output = model(input_data + perturbation[:input_data.size(0)])
            output_probs = torch.nn.Softmax(dim=1)(model_output)
            feature_probs = torch.nn.Softmax(dim=1)(feature_output)

            loss_function = torch.nn.MSELoss(reduce=True, size_average=True)
            smoothed_targets = Variable(torch.tensor(compute_smoothed_labels(target_labels, config.factor)).cuda())

            loss_value = compute_smoothed_loss(model_output, smoothed_targets.float()) + config.lamda * (loss_function(output_probs.float(), adv_output.float())+loss_function(feature_probs.float(), original_features.float()))/(loss_function((input_data + perturbation[:input_data.size(0)]).float(), (input_data).float())+0.125)

            optimizer.zero_grad()
            loss_value.backward()

            # Apply Max-Norm Regularization
            for param_name, param_val in model.named_parameters():
                if 'weight' in param_name:
                    torch.nn.utils.clip_grad_norm_(param_val, config.max_norm)

            optimizer.step()

            total_loss += loss_value.item() * target_labels.size(0)
            total_accuracy += (model_output.max(1)[1] == target_labels).sum().item()
            robust_accuracy = (output_probs.max(1)[1] == target_labels).sum().item()
            clean_accuracy = (adv_output.max(1)[1] == target_labels).sum().item()
            samples_processed += target_labels.size(0)

            if robust_accuracy / clean_accuracy < config.EMA_value:
                teacher_model.update_parameters(model)
                teacher_model.apply_shadow_weights()

            lr_scheduler.step()
            batch_end = time.time()
            epoch_duration += batch_end - batch_start

        initial_loss.append(initial_batch_loss / samples_processed)
        initial_accuracy.append(initial_batch_accuracy / samples_processed)
        training_loss.append(total_loss / samples_processed)
        training_accuracy.append(total_accuracy / samples_processed)

        current_lr = lr_scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    epoch_index, epoch_duration, current_lr, total_loss / samples_processed, total_accuracy / samples_processed)

        logger.info('==> Building model..')
        if config.model == "VGG":
            evaluation_model = VGG('VGG19').cuda()
        elif config.model == "ResNet18":
            evaluation_model = ResNet18().cuda()
        elif config.model == "PreActResNest18":
            evaluation_model = PreActResNet18().cuda()
        elif config.model == "WideResNet":
            evaluation_model = WideResNet().cuda()
        teacher_model = ExponentialMovingAverage(model)
        evaluation_model.load_state_dict(teacher_model.model.state_dict())
        evaluation_model.float()
        evaluation_model.eval()

        eval_epsilon = (config.epsilon / 255.) / std
        pgd_loss, pgd_accuracy = evaluate_powerful_pgd(test_loader, evaluation_model, 10, 1, eval_epsilon)
        test_loss, test_accuracy = evaluate_standard(test_loader, evaluation_model)

        epoch_clean_list.append(test_accuracy)
        epoch_pgd_list.append(pgd_accuracy)

        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_accuracy, pgd_loss, pgd_accuracy)

        if best_accuracy <= pgd_accuracy:
            best_accuracy = pgd_accuracy
            torch.save(evaluation_model.state_dict(), os.path.join(output_directory, 'best_model.pth'))


    torch.save(evaluation_model.state_dict(), os.path.join(output_directory, 'final_model.pth'))
    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)
    print(epoch_clean_list)
    print(epoch_pgd_list)

if __name__ == "__main__":
    execute_training()
