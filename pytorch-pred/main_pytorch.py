import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_generator import DataGenerator, TestDataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       calculate_confusion_matrix, calculate_accuracy, 
                       plot_confusion_matrix, print_accuracy, 
                       write_leaderboard_submission, write_evaluation_submission)
from models_pytorch import move_data_to_gpu, DecisionLevelMaxPooling, DecisionLevelAvgPooling, DecisionLevelSingleAttention, DecisionLevelFlatten, \
    	CnnNoPooling_Max, CnnNoPooling_Avg, CnnNoPooling_roi, CnnNoPooling_Attention, CnnNoPooling_roi_attention,\
        CnnAtrous_Max, CnnAtrous_Avg, CnnAtrous_roi, CnnAtrous_Attention, CnnAtrous_roi_attention
import config
from torch.autograd import Variable

Model = CnnAtrous_Attention
batch_size = 16
cond_layer = 4

def evaluate(model, generator, data_type, devices, max_iteration, cuda):
    """Evaluate
    
    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
      max_iteration: int, maximum iteration for validation
      cuda: bool.
      
    Returns:
      accuracy: float
    """
    
    # Generate function
    generate_func = generator.generate_validate(data_type=data_type, 
                                                devices=devices, 
                                                shuffle=True, 
                                                max_iteration=max_iteration)
            
    # Forward
    dict = forward(model=model, 
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_target=True)

    outputs_device = dict['output_device']    # (audios_num, classes_num)
    devices = dict['device']    # (audios_num, classes_num)
    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)
    
    predictions = np.argmax(outputs, axis=-1)   # (audios_num,)
    predictions_device = np.argmax(outputs_device, axis=-1)   # (audios_num,)

    # Evaluate
    classes_num = outputs.shape[-1]
    devices_num = outputs_device.shape[-1]

    loss = F.nll_loss(Variable(torch.Tensor(outputs)), Variable(torch.LongTensor(targets))).data.numpy()
    loss = float(loss)

    loss_device = F.nll_loss(Variable(torch.Tensor(outputs_device)), Variable(torch.LongTensor(devices))).data.numpy()
    loss_device = float(loss_device)
    
    confusion_matrix = calculate_confusion_matrix(
        targets, predictions, classes_num)
    
    accuracy = calculate_accuracy(targets, predictions, classes_num, 
                                  average='macro')
    accuracy_device = calculate_accuracy(devices, predictions_device, devices_num, 
                                  average='macro')

    return accuracy, loss, accuracy_device, loss_device

# forward: model_pytorch---        Return_heatmap = False
# forward_heatmap: model_pytorch---        Return_heatmap = True
def forward(model, generate_func, cuda, return_target):
    """Forward data to a model.
    
    Args:
      generate_func: generate function
      cuda: bool
      return_target: bool
      
    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """
    
    outputs_device = []
    devices = []
    outputs = []
    audio_names = []
    
    if return_target:
        targets = []
    
    # Evaluate on mini-batch
    for data in generate_func:
            
        if return_target:
            (batch_x, batch_y, batch_audio_names, batch_device) = data
            
        else:
            (batch_x, batch_audio_names) = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        model.eval()
        batch_output, batch_output_device = model(batch_x)

        # Append data
        outputs_device.append(batch_output_device.data.cpu().numpy())
        devices.append(batch_device)
        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)
        
        if return_target:
            targets.append(batch_y)

    dict = {}

    outputs_device = np.concatenate(outputs_device, axis=0)
    dict['output_device'] = outputs_device

    devices = np.concatenate(devices, axis=0)
    dict['device'] = devices

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs
    
    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names
    
    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets
        
    return dict


def forward_heatmap(model, generate_func, cuda, has_target):
    """Forward data to a model.
    
    Args:
      generate_func: generate function
      cuda: bool
      has_target: bool
      
    Returns:
      (outputs, targets, audio_names) | (outputs, audio_names)
    """

    model.eval()

    outputs = []
    targets = []
    audio_names = []
    outputs_heatmap = [] ###############################

    # Evaluate on mini-batch
    for data in generate_func:
            
        if has_target:
            (batch_x, batch_y, batch_audio_names) = data
            targets.append(batch_y)
            
        else:
            (batch_x, batch_audio_names) = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        batch_output, batch_heatmap = model(batch_x)###########################################

        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)
        outputs_heatmap.append(batch_heatmap.data.cpu().numpy())#######################################

    outputs = np.concatenate(outputs, axis=0)
    audio_names = np.concatenate(audio_names, axis=0)
    outputs_heatmap = np.concatenate(outputs_heatmap, axis=0)###########################################
    
    if has_target:
        targets = np.concatenate(targets, axis=0)
        return outputs, targets, audio_names, outputs_heatmap############################################
    else:
        return outputs, audio_names


def train(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    filename = args.filename
    validate = args.validate
    holdout_fold = args.holdout_fold
    mini_data = args.mini_data
    cuda = args.cuda

    labels = config.labels

    if 'mobile' in subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    classes_num = len(labels)
    devices_num = len(devices)

    # Paths
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 'mini_development.h5')
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 'development.h5')

    if validate:
        
        dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                     'fold{}_train.csv'.format(holdout_fold))
                                    
        dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                    'fold{}_evaluate.csv'.format(holdout_fold))
                              
        models_dir = os.path.join(workspace, 'models', subdir, filename,
                                  'holdout_fold={}'.format(holdout_fold))
                                        
    else:
        dev_train_csv = None
        dev_validate_csv = None
        
        models_dir = os.path.join(workspace, 'models', subdir, filename,
                                  'full_train')

    create_folder(models_dir)

    # Model
    model = Model(classes_num, devices_num, cond_layer)

    if cuda:
        model.cuda()

    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv)

    # Optimizer
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)

    train_bgn_time = time.time()

    # Train on mini batches
    for (iteration, (batch_x, batch_y, batch_device)) in enumerate(generator.generate_train()):

        # Evaluate
        if iteration % 100 == 0:

            train_fin_time = time.time()

            (tr_acc, tr_loss, tr_acc_device, tr_loss_device) = evaluate(model=model,
                                         generator=generator,
                                         data_type='train',
                                         devices=devices,
                                         max_iteration=None,
                                         cuda=cuda)

            logging.info('tr_acc: {:.3f}, tr_loss: {:.3f}, tr_acc_device: {:.3f}, tr_loss_device: {:.3f}'.format(
                tr_acc, tr_loss, tr_acc_device, tr_loss_device))

            if validate:

                (va_acc, va_loss, va_acc_device, va_loss_device) = evaluate(model=model,
                                             generator=generator,
                                             data_type='validate',
                                             devices=devices,
                                             max_iteration=None,
                                             cuda=cuda)

                logging.info('va_acc: {:.3f}, va_loss: {:.3f}, va_acc_device: {:.3f}, va_loss_device: {:.3f}'.format(
                    va_acc, va_loss, va_acc_device, tr_loss_device))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

        # Save model
        if iteration % 1000 == 0 and iteration > 0:

            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))
            
        # Reduce learning rate
        if iteration % 200 == 0 > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        # Train
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        batch_device = move_data_to_gpu(batch_device, cuda)

        model.train()
        batch_output, batch_output_device = model(batch_x)

        loss = F.nll_loss(batch_output, batch_y)
        loss_device = F.nll_loss(batch_output_device, batch_device)

	if va_acc_device>=0.98:
            lambda_w = 0.0001
        else:
            lambda_w = 1
        loss = loss + lambda_w * loss_device

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == 15000:
            break


def inference_validation_data(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda

    labels = config.labels

    if 'mobile' in subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    validation = True
    classes_num = len(labels)
    devices_num = len(devices)

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                             'development.h5')

    dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                 'fold1_train.csv')
                                 
    dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                    'fold{}_evaluate.csv'.format(holdout_fold))

    model_path = os.path.join(workspace, 'models', subdir, filename,
                              'holdout_fold={}'.format(holdout_fold),
                              'md_{}_iters.tar'.format(iteration))

    # Load model
    model = Model(classes_num, devices_num, cond_layer)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Predict & evaluate
    for device in devices:

        print('Device: {}'.format(device))

        # Data generator
        generator = DataGenerator(hdf5_path=hdf5_path,
                                  batch_size=batch_size,
                                  dev_train_csv=dev_train_csv,
                                  dev_validate_csv=dev_validate_csv)

        generate_func = generator.generate_validate(data_type='validate', 
                                                     devices=device, 
                                                     shuffle=False)

        # Inference
        dict = forward(model=model,
                       generate_func=generate_func, 
                       cuda=cuda, 
                       return_target=True)

        outputs = dict['output']    # (audios_num, classes_num)
        targets = dict['target']    # (audios_num, classes_num)

#	(outputs, targets, audio_names, outputs_heatmap) = forward_heatmap(model=model,###################
#                                   generate_func=generate_func, 
#                                   cuda=cuda, 
#                                   has_target=True)

        predictions = np.argmax(outputs, axis=-1)

        classes_num = outputs.shape[-1]

##########################################################################
#        heatmaps = []
#        classes = []
#        for i in range(0, len(predictions)):
#            pred_num = predictions[i]
#            if pred_num == targets[i]:
#                if not (pred_num in classes):
#                    classes.append(pred_num)
#                    print 'classes:'
#                    print classes
#                    logging.info('\n')
#                    logging.info(outputs_heatmap[i][pred_num])
#                    logging.info('class num: ')
#                    logging.info(pred_num)
#                    heatmaps.append(outputs_heatmap[i][pred_num])
                    
############################################################################        

        # Evaluate
        confusion_matrix = calculate_confusion_matrix(targets, predictions, classes_num)
            
        class_wise_accuracy = calculate_accuracy(targets, predictions, classes_num)


        # save
        # np.save(os.path.join(workspace, 'logs', 'main_pytorch', str(device)+"heatmap.npy"),heatmaps)##############
        # np.save(os.path.join(workspace, 'logs', 'main_pytorch', str(device)+"confusionMat.npy"),confusion_matrix)#########

        # Print
        print_accuracy(class_wise_accuracy, labels)
        print('confusion_matrix: \n', confusion_matrix)
        logging.info('confusion_matrix: \n', confusion_matrix)

        # Plot confusion matrix
#        plot_confusion_matrix(
#            confusion_matrix,
#            title='Device {}'.format(device.upper()), 
#            labels=labels,
#            values=class_wise_accuracy,
#            path=os.path.join(workspace, 'logs', 'main_pytorch', 'fig-confmat-device-'+device+'.pdf'))
    
            

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--subdir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--holdout_fold', type=int)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    
    parser_inference_validation_data = subparsers.add_parser('inference_validation_data')
    parser_inference_validation_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation_data.add_argument('--subdir', type=str, required=True)
    parser_inference_validation_data.add_argument('--workspace', type=str, required=True)
    parser_inference_validation_data.add_argument('--holdout_fold', type=int, required=True)
    parser_inference_validation_data.add_argument('--iteration', type=int, required=True)
    parser_inference_validation_data.add_argument('--cuda', action='store_true', default=False)
                     

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation_data':
        inference_validation_data(args)

    else:
        raise Exception('Error argument!')

