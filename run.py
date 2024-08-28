from torchvision import transforms
from dataset import TorchDataset

from options import Options
import decision
import experiment
import experiment_nm
import datatransforms
import utils
import os
import argparse
### COMMAND LINE ARGUEMNTS ###
parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=str, choices=['train', 'test', 'both'])
parser.add_argument('--data-path', type=str) 
parser.add_argument('--batch-size', type=int, default = 2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--output-dir', type=str, default=None)
parser.add_argument('--NM', dest='nm', action='store_true')
parser.set_defaults(nm=False)
parser.add_argument('--loss-function', type=str, default='hist')
parser.add_argument('--joint-learning', dest='joint_learning', action='store_true')
parser.set_defaults(joint_learning=False)


args = parser.parse_args()
resume_from_epoch = None
### CONFIGURATION ###
opts = Options()
# Setting options
data_dir = args.data_path
opts.datapath = args.data_path
opts.lr = args.lr
opts.batch_size = args.batch_size
if args.output_dir is not None:
    opts.experiment_name = args.output_dir
# List the names of the trajectories in the dataset
base_path = opts.datapath
opts.loss_function = args.loss_function    
if os.path.exists(opts.datapath + '/trainingdata.mat'):
    print('\n*** Training and test .mat data already exist. ***\n')
    mat_file_train = opts.datapath + '/trainingdata.mat'
    mat_file_test = opts.datapath + '/evaldata.mat'
    mat_file_val = opts.datapath + '/evaldata.mat'
opts.joint_learning = args.joint_learning
opts.train_epochs = args.epoch
print(opts)

# Create data transforms
data_transforms = transforms.Compose([
                datatransforms.ToTensor(opts),
                ])
training_dataset = TorchDataset(data_dir, mat_file_train, opts, data_transforms)
validation_dataset = TorchDataset(data_dir, mat_file_val, opts, data_transforms)
testing_dataset = TorchDataset(data_dir, mat_file_test, opts, data_transforms)

print('Running {} training...'.format(opts.parameter))
model = decision.ExposureStrategyNet(nm_flag = args.nm, joint_learning_flag = args.joint_learning)        
model.to(opts.device)

print('\n{:-^50}'.format(' Network Initialized '))
utils.print_network(model)
print('-' * 50 + '\n')

if args.stage == 'train' or args.stage == 'both':
    print(opts)
    if args.nm:
        experiment_nm.train(opts, model, training_dataset, validation_dataset, opts.train_epochs, resume_from_epoch=resume_from_epoch)
    else:
        experiment.train(opts, model, training_dataset, validation_dataset, opts.train_epochs, resume_from_epoch=resume_from_epoch)
        
    
if args.stage == 'test' or args.stage == 'both':
    print('\n{:-^50}'.format(' Testing Model '))
    if args.nm:
        experiment_nm.test(opts, model, testing_dataset, save_loss=True)
    else:
        experiment.test(opts, model, testing_dataset, save_loss=True)
    

print('-' * 50 + '\n')
print('Training Complete!')
print('-' * 50 + '\n')