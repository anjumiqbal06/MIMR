import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
import argparse
import numpy as np
import copy
import torch
from torchvision import datasets, transforms
import os
import time
from cutout import Cutout
from ImageNet_models import *
from utils_2 import *
from Feature_model.feature_preact_resnet import *
import logging
logger = logging.getLogger(__name__)
from TinyImageNet import TinyImageNet
import matplotlib.pyplot as plt

def parse_configuration():
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument('--data-dir', default='tiny_imagenet/tiny-imagenet-200', type=str)
    config_parser.add_argument('--epochs', default=110, type=int)
    config_parser.add_argument('--batch-size', default=128, type=int)
    config_parser.add_argument('--epsilon', default=8, type=int)
    config_parser.add_argument('--weight-decay', default=5e-4, type=float)
    config_parser.add_argument('--momentum', default=0.9, type=float)
    config_parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    config_parser.add_argument('--lr-min', default=0., type=float)
    config_parser.add_argument('--lr-max', default=0.1, type=float)
    config_parser.add_argument('--model', default='PreActResNest18', type=str, help='model name')
    config_parser.add_argument('--c_num', default=0.125, type=float)
    config_parser.add_argument('--EMA_value', default=0.55, type=float)
    config_parser.add_argument('--alpha', default=8, type=float, help='Step size')
    config_parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    config_parser.add_argument('--out_dir', default='output', type=str, help='Output directory')
    config_parser.add_argument('--seed', default=0, type=int, help='Random seed')
    config_parser.add_argument('--lamda', default=38, type=float, help='Label Smoothing')
    config_parser.add_argument('--normal_mean', default=0, type=float, help='normal_mean')
    config_parser.add_argument('--normal_std', default=1, type=float, help='normal_std')
    config_parser.add_argument('--factor', default=0.3, type=float)
    config_parser.add_argument('--length', type=int, default=6, help='length of the holes')
    config_parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
    config_parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    config_parser.add_argument('--max-norm', default=1.0, type=float, help='Maximum norm value for regularization')
    return config_parser.parse_args()

config = parse_configuration()

def New_ImageNet_get_loaders_64(dir_, batch_size):
    train_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
    ])
    train_transforms.transforms.append(Cutout(n_holes=config.n_holes, length=config.length))
    test_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
    ])
    train_dataset = TinyImageNet(dir_, 'train', transform=train_transforms, in_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    test_dataset = TinyImageNet(dir_, 'val', transform=test_transforms, in_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader

import numpy as np
from torch.autograd import Variable

def compute_smoothed_labels(target_labels, smoothing_factor):
    one_hot_encoded = np.eye(200)[target_labels.cuda().data.cpu().numpy()]
    smoothed_labels = one_hot_encoded * smoothing_factor + (one_hot_encoded - 1.) * ((smoothing_factor - 1) / float(200 - 1))
    return smoothed_labels

def compute_smoothed_loss(model_output, target):
    log_probs = F.log_softmax(model_output, dim=-1)
    loss_value = (-target * log_probs).sum(dim=-1).mean()
    return loss_value

class ExponentialMovingAverage(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
        self.step_counter = 0
        self.model = copy.deepcopy(model)
        self.alpha_value = alpha
        self.buffer_ema_enabled = buffer_ema
        self.shadow_weights = self.get_model_state()
        self.backup_weights = {}
        self.parameter_names = [k for k, _ in self.model.named_parameters()]
        self.buffer_names = [k for k, _ in self.model.named_buffers()]

    def update_params(self, model):
        decay_factor = min(self.alpha_value, (self.step_counter + 1) / (self.step_counter + 10))
        model_state = model.state_dict()
        for name in self.parameter_names:
            self.shadow_weights[name].copy_(
                decay_factor * self.shadow_weights[name] + (1 - decay_factor) * model_state[name])
        for name in self.buffer_names:
            if self.buffer_ema_enabled:
                self.shadow_weights[name].copy_(
                    decay_factor * self.shadow_weights[name] + (1 - decay_factor) * model_state[name])
            else:
                self.shadow_weights[name].copy_(model_state[name])
        self.step_counter += 1

    def apply_shadow(self):
        self.backup_weights = self.get_model_state()
        self.model.load_state_dict(self.shadow_weights)

    def restore(self):
        self.model.load_state_dict(self.backup_weights)

    def get_model_state(self):
        return {k: v.clone().detach() for k, v in self.model.state_dict().items()}

def run_training():
    configuration = parse_configuration()
    output_directory = os.path.join(configuration.out_dir, 'Tiny_ImageNet')
    output_directory = os.path.join(output_directory, 'c_num_' + str(configuration.c_num))
    output_directory = os.path.join(output_directory, 'model_' + str(configuration.model))
    output_directory = os.path.join(output_directory, 'factor_' + str(configuration.factor))
    output_directory = os.path.join(output_directory, 'length_' + str(configuration.length))
    output_directory = os.path.join(output_directory, 'EMA_value_' + str(configuration.EMA_value))
    output_directory = os.path.join(output_directory, 'lamda_' + str(configuration.lamda))

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
    logger.info(configuration)

    np.random.seed(configuration.seed)
    torch.manual_seed(configuration.seed)
    torch.cuda.manual_seed(configuration.seed)

    train_loader, test_loader = New_ImageNet_get_loaders_64(configuration.data_dir, configuration.batch_size)

    eps_value = (configuration.epsilon / 255.) / std
    alpha_value = (configuration.alpha / 255.) / std

    print('==> Building model..')
    logger.info('==> Building model..')
    if configuration.model == "VGG":
        model = VGG('VGG19')
    elif configuration.model == "ResNet18":
        model = ResNet18()
    elif configuration.model == "PreActResNest18":
        model = Feature_PreActResNet18()
    elif configuration.model == "WideResNet":
        model = WideResNet()
    model = model.cuda()
    model.train()
    teacher_model = ExponentialMovingAverage(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=configuration.lr_max, momentum=configuration.momentum, weight_decay=configuration.weight_decay)

    if configuration.delta_init == 'previous':
        perturbation = torch.zeros(configuration.batch_size, 3, 32, 32).cuda()

    momentum_buff = torch.zeros_like(perturbation).cuda()

    total_steps = configuration.epochs * len(train_loader)
    if configuration.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=configuration.lr_min, max_lr=configuration.lr_max,
                                                      step_size_up=total_steps / 2, step_size_down=total_steps / 2)
    elif configuration.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[total_steps * 99 / 110, total_steps * 104 / 110],
                                                         gamma=0.1)

    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    best_accuracy = 0
    epoch_train_clean_metrics = []
    epoch_train_pgd_metrics = []
    epoch_clean_metrics = []
    epoch_pgd_metrics = []
    initial_losses = []
    initial_accuracies = []
    training_losses = []
    training_accuracies = []
    for epoch_index in range(configuration.epochs):
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

            if configuration.delta_init != 'previous':
                perturbation = torch.zeros_like(input_data).cuda()
            if configuration.delta_init == 'random':
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

            initial_batch_loss += loss_value.item() * target_labels.size(0)
            initial_batch_accuracy += (adv_output.max(1)[1] == target_labels).sum().item()

            loss_value.backward(retain_graph=True)
            perturbation_gradient = perturbation.grad.detach()
            perturbation.data = clamp(perturbation + alpha_value * torch.sign(perturbation_gradient), -eps_value, eps_value)
            perturbation.data[:input_data.size(0)] = clamp(perturbation[:input_data.size(0)], lower_limit - input_data, upper_limit - input_data)

            momentum_buff = configuration.momentum * momentum_buff + (1 - configuration.momentum) * perturbation_gradient
            perturbation = perturbation + alpha_value * momentum_buff

            perturbation.data = clamp(perturbation, lower_limit - input_data, upper_limit - input_data)
            perturbation = perturbation.detach()

            model_output, feature_output = model(input_data + perturbation[:input_data.size(0)])
            output_probs = torch.nn.Softmax(dim=1)(model_output)
            feature_probs = torch.nn.Softmax(dim=1)(feature_output)

            mse_loss = torch.nn.MSELoss(reduce=True, size_average=True)
            smoothed_targets = Variable(torch.tensor(compute_smoothed_labels(target_labels, configuration.factor)).cuda())

            loss_value = compute_smoothed_loss(model_output, smoothed_targets.float()) + configuration.lamda * (mse_loss(output_probs.float(), adv_output.float()) + mse_loss(feature_probs.float(), original_features.float())) / (mse_loss((input_data + perturbation[:input_data.size(0)]).float(), (input_data).float()) + 0.125)

            optimizer.zero_grad()
            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), configuration.max_norm)

            optimizer.step()
            total_loss += loss_value.item() * target_labels.size(0)
            total_accuracy += (model_output.max(1)[1] == target_labels).sum().item()
            robust_accuracy = (output_probs.max(1)[1] == target_labels).sum().item()
            clean_accuracy = (adv_output.max(1)[1] == target_labels).sum().item()
            samples_processed += target_labels.size(0)

            if robust_accuracy / clean_accuracy < configuration.EMA_value:
                teacher_model.update_params(model)
                teacher_model.apply_shadow()

            scheduler.step()
            batch_end = time.time()
            epoch_duration += batch_end - batch_start

        initial_losses.append(initial_batch_loss / samples_processed)
        initial_accuracies.append(initial_batch_accuracy / samples_processed)
        training_losses.append(total_loss / samples_processed)
        training_accuracies.append(total_accuracy / samples_processed)

        current_lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    epoch_index, epoch_duration, current_lr, total_loss / samples_processed, total_accuracy / samples_processed)

        eval_epsilon = (configuration.epsilon / 255.) / std
        pgd_loss, pgd_accuracy = evaluate_pgd(test_loader, model, 10, 1)
        test_loss, test_accuracy = evaluate_standard(test_loader, model)
        epoch_clean_metrics.append(test_accuracy)
        epoch_pgd_metrics.append(pgd_accuracy)

        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_accuracy, pgd_loss, pgd_accuracy)

        if best_accuracy <= pgd_accuracy:
            best_accuracy = pgd_accuracy
            torch.save(model.state_dict(), os.path.join(output_directory, 'best_model.pth'))

        # Save plot
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_clean_metrics, label='Test Accuracy')
        plt.plot(epoch_pgd_metrics, label='PGD Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Test and PGD Test Accuracy over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_directory, f'accuracy_epoch_{epoch_index}.png'))
        plt.close()

    torch.save(model.state_dict(), os.path.join(output_directory, 'final_model.pth'))
    logger.info(epoch_clean_metrics)
    logger.info(epoch_pgd_metrics)
    print(epoch_clean_metrics)
    print(epoch_pgd_metrics)

if __name__ == "__main__":
    run_training()
