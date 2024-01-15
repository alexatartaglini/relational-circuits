from torchvision import models, transforms
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig, CLIPVisionModelWithProjection, AutoProcessor, CLIPConfig
#import clip
from torch.utils.data import DataLoader
from data import SameDifferentDataset, call_create_stimuli
import torch.nn as nn
import torch
import argparse
import os
import shutil
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb
import numpy as np
import sys
from math import floor
import copy
import pickle


os.chdir(sys.path[0])

def train_model(args, model, device, data_loader, dataset_size, optimizer,
                scheduler, log_dir, val_datasets, val_dataloaders, 
                test_table, val_labels=None):
    
    if not val_labels:
        val_labels = list(range(len(val_dataloaders)))
        
    aug_string = ''
    if args.rotation:
        aug_string += 'R'
    if args.scaling:
        aug_string += 'S'
    if len(aug_string) == 0:
        aug_string = 'N'
    
    int_to_label = {0: 'different', 1: 'same'}
    
    model_type = args.model_type
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    save_model_freq = args.save_model_freq
    log_preds_freq = args.log_preds_freq
    
    if save_model_freq == -1:
        save_model_epochs = [num_epochs - 1]
    else:
        save_model_epochs = np.linspace(0, num_epochs, save_model_freq, dtype=int)
    
    if log_preds_freq > 0:
        log_preds_epochs = np.linspace(0, num_epochs, log_preds_freq, dtype=int)
    else:
        log_preds_epochs = [-1]

    criterion = nn.CrossEntropyLoss()
    
    if args.feature_extract:
        features = {}  # Keep track of features
        backbone = model['backbone'].to(device)
        model = model['classifier']
        print('getting features...')

        if args.model_type == 'resnet':
            model_string = 'resnet_{0}'.format(args.cnn_size)
        elif args.model_type == 'vit':
            model_string = 'vit_b{0}'.format(args.patch_size)
        else:              
            if 'vit' in args.model_type:
                model_string = 'clip_vit_b{0}'.format(args.patch_size)
            else:
                model_string = 'clip_resnet50'
        
        features = pickle.load(open(f'features/{model_string}_{aug_string}.pickle', 'rb'))
        in_features = list(model.children())[0].in_features
        
        labels_to_labels = {}
    else:
        model = model['classifier']

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        model.train()

        running_loss = 0.0
        running_acc = 0.0

        # Iterate over data.
        for bi, (d, f) in enumerate(data_loader):
            if args.feature_extract:
                inputs = torch.zeros((batch_size, in_features)).to(device)
                for fi in range(len(f)):
                    try:
                        inputs[fi, :] = features[f[fi]].to(device)
                    except KeyError:
                        if 'vit' in model_type:
                            inputs = d['pixel_values'].squeeze(1).to(device)
                        else:
                            inputs = d['image'].to(device)
                        inputs[fi, :] = backbone(inputs)[fi, :]
                        features[f[fi]] = backbone(inputs)[fi, :]
            else:
                if 'vit' in model_type:
                    inputs = d['pixel_values'].squeeze(1).to(device)
                else:
                    inputs = d['image'].to(device)
                
            labels = d['label'].to(device)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                outputs = model(inputs)

                if 'vit' in model_type:
                    if "clip" in model_type:
                        outputs = outputs.image_embeds
                    else:
                        outputs = outputs.logits

                loss = criterion(outputs, labels)
                acc = accuracy_score(labels.to('cpu'), outputs.to('cpu').argmax(1))

                loss.backward()
                optimizer.step()

            print('\t({0}/{1}) Batch loss: {2:.4f}'.format(bi + 1, dataset_size // batch_size, loss.item()))
            running_loss += loss.detach().item() * inputs.size(0)
            running_acc += acc * inputs.size(0)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_acc / dataset_size
        print('Epoch loss: {:.4f}'.format(epoch_loss))
        print('Epoch accuracy: {:.4f}'.format(epoch_acc))
        print()

        metric_dict = {'epoch': epoch, 'loss': epoch_loss, 'acc': epoch_acc,
                       'lr': optimizer.param_groups[0]['lr']}

        # Save the model
        if epoch in save_model_epochs and args.checkpoint:
            torch.save(model.state_dict(), f'{log_dir}/model_{epoch}_{lr}_{wandb.run.id}.pth')

        # Perform evaluations
        model.eval()
        for i in range(len(val_dataloaders)):
            val_dataloader = val_dataloaders[i]
            val_dataset = val_datasets[i]
            val_label = val_labels[i]

            with torch.no_grad():
                running_loss_val = 0.0
                running_acc_val = 0.0
                running_roc_auc = 0.0

                for bi, (d, f) in enumerate(val_dataloader):
                    if 'vit' in model_type:
                        inputs = d['pixel_values'].squeeze(1).to(device)
                    else:
                        inputs = d['image'].to(device)
                    
                    if args.feature_extract:
                        inputs = torch.zeros((inputs.shape[0], in_features)).to(device)
                        for fi in range(len(f)):
                            inputs[fi, :] = features[f[fi]].to(device)
                    
                    labels = d['label'].to(device)

                    outputs = model(inputs)
                    if 'vit' in model_type:
                        if "clip" in model_type:
                            outputs = outputs.image_embeds
                        else:
                            outputs = outputs.logits

                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(1)
                    acc = accuracy_score(labels.to('cpu'), preds.to('cpu'))
                    roc_auc = roc_auc_score(labels.to('cpu'), outputs.to('cpu')[:, -1])
                    
                    # Log error examples
                    if epoch in log_preds_epochs and not args.feature_extract:
                        error_idx = (labels + preds == 1).cpu()
                        error_ims = inputs[error_idx, :, :, :]
                        error_paths = [name.split('/')[-1] for name in np.asarray(list(f), dtype=object)[error_idx]]
                        error_preds = [int_to_label[p.item()] for p in preds[error_idx]]
                        error_truths = [int_to_label[l.item()] for l in labels[error_idx]]
                        same_scores = outputs[error_idx, 0]
                        diff_scores = outputs[error_idx, 1]
                        same_acc = len(labels[labels + preds == 2]) / len(labels[labels == 1])
                        diff_acc = len(labels[labels + preds == 0]) / len(labels[labels == 0]) if len(labels[labels==0]) > 0 else 0
                        for j in range(len(same_scores)):
                            test_table.add_data(epoch, error_paths[j], wandb.Image(error_ims[j, :, :, :]),
                                                val_label, error_preds[j], error_truths[j], same_scores[j], 
                                                diff_scores[j], same_acc, diff_acc)

                    running_acc_val += acc * inputs.size(0)
                    running_loss_val += loss.detach().item() * inputs.size(0)
                    running_roc_auc += roc_auc * inputs.size(0)

                epoch_loss_val = running_loss_val / len(val_dataset)
                epoch_acc_val = running_acc_val / len(val_dataset)
                epoch_roc_auc = running_roc_auc / len(val_dataset)

                print()
                print('Validation: {0}'.format(val_label))
                print('Val loss: {:.4f}'.format(epoch_loss_val))
                print('Val acc: {:.4f}'.format(epoch_acc_val))
                print('Val ROC-AUC: {:.4f}'.format(epoch_roc_auc))
                print()

                metric_dict['val_loss_{0}'.format(val_label)] = epoch_loss_val
                metric_dict['val_acc_{0}'.format(val_label)] = epoch_acc_val
                metric_dict['val_roc_auc_{0}'.format(val_label)] = epoch_roc_auc
           
        if epoch in log_preds_epochs:
            try:
                test_data_at = wandb.Artifact(f'test_errors_{run_id}_{epoch}', type='predictions')
                test_data_at.add(test_table, 'predictions')
                wandb.run.log_artifact(test_data_at).wait() 
            except OSError:
                try:
                    shutil.rmtree(args.wandb_cache_dir) 
                    test_data_at = wandb.Artifact(f'test_errors_{run_id}_{epoch}', type='predictions')
                    test_data_at.add(test_table, 'predictions')
                    wandb.run.log_artifact(test_data_at).wait() 
                except OSError:
                    pass
                except TypeError:
                    pass
            
        if scheduler:
            scheduler.step(metric_dict[f'val_acc_{val_labels[0]}'])  # Reduce LR based on validation accuracy

        # Log metrics
        wandb.log(metric_dict)

    return model

# Set device
try:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
except AttributeError:  # if MPS is not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--wandb_proj', type=str, default='same-different-transformers',
                    help='Name of WandB project to store the run in.')
parser.add_argument('--wandb_entity', type=str, default=None, help='Team to send run to.')
parser.add_argument('--num_gpus', type=int, help='number of available GPUs.', default=1)

# Model/architecture arguments
parser.add_argument('-m', '--model_type', help='Model to train: resnet, vit, clip_rn, clip_vit.',
                    type=str, required=True)
parser.add_argument('--patch_size', type=int, default=32, help='Size of patch (eg. 16 or 32).')
parser.add_argument('--cnn_size', type=int, default=50,
                    help='Number of layers for ResNet (eg. 50 = ResNet-50).', required=False)
parser.add_argument('--feature_extract', action='store_true', default=False,
                    help='Only train the final layer; freeze all other layers.')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Use ImageNet pretrained models. If false, models are trained from scratch.')

# Training arguments
parser.add_argument('-td', '--train_datasets', nargs='+', required=False, 
                    help='Names of all stimulus subdirectories to draw train stimuli from.', 
                        default=['NOISE'])
parser.add_argument('-vd','--val_datasets', nargs='+', required=False, 
                    default='all',
                    help='Names of all stimulus subdirectories to draw validation datasets from. \
                        Input "all" in order to test on all existing sets.')
parser.add_argument('--optim', type=str, default='adamw',
                    help='Training optimizer, eg. adam, adamw, sgd.')
parser.add_argument('--lr', type=float, default=2e-6, help='Learning rate.')
parser.add_argument('--lr_scheduler', default='reduce_on_plateau', help='LR scheduler.')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=64, help='Train/validation batch size.')
parser.add_argument('--seed', type=int, default=-1, help='If not given, picks random seed.')

# Stimulus arguments
parser.add_argument('--k', type=int, default=2, help='Number of objects per scene.')
parser.add_argument('--rotation', action='store_true', default=False,
                    help='Randomly rotate the objects in the stimuli.')
parser.add_argument('--scaling', action='store_true', default=False,
                    help='Randomly scale the objects in the stimuli.')
parser.add_argument('--unaligned', action='store_true', default=True,
                    help='Misalign the objects from ViT patches.')
parser.add_argument('--multiplier', type=int, default=1, help='Factor by which to scale up \
                    stimulus size. For example, if patch_size=32 and multiplier=2, the \
                    stimuli will be 64x64.')

# Dataset size arguments
parser.add_argument('--ood', action='store_true', default=False, help='use datasets with unfamiliar test/val objs')
parser.add_argument('--n_train', type=int, default=6400, help='Size of training dataset to use.')
parser.add_argument('--n_train_tokens', type=int, default=-1, help='Number of unique tokens to use \
                    in the training dataset. If -1, then the maximum number of tokens is used.')
parser.add_argument('--n_val_tokens', type=int, default=-1, help='Number of unique tokens to use \
                    in the validation dataset. If -1, then number tokens = (total - n_train_tokens) // 2.')
parser.add_argument('--n_test_tokens', type=int, default=-1, help='Number of unique tokens to use \
                    in the test dataset. If -1, then number tokens = (total - n_train_tokens) // 2.')
parser.add_argument('--n_val', type=int, default=-1,
                    help='Total # validation stimuli. Default: equal to n_train.')
parser.add_argument('--n_test', type=int, default=-1,
                    help='Total # test stimuli. Default: equal to n_train.')
parser.add_argument('--n_train_ood', nargs='+', required=False, default=[],
                    help='Size of OOD training sets.')
parser.add_argument('--n_train_tokens_ood', nargs='+', required=False, default=[],
                    help='Number of unique tokens in OOD training sets. Default: n_train_tokens.')
parser.add_argument('--n_val_ood', nargs='+', required=False, default=[],
                    help='Size of OOD validation sets. Default: equal to n_train_ood.')
parser.add_argument('--n_val_tokens_ood', nargs='+', required=False, default=[],
                    help='Number of unique tokens in OOD validation sets. Default: n_val_tokens.')
parser.add_argument('--n_test_ood', nargs='+', required=False, default=[],
                    help='Size of OOD test sets. Default: equal to n_train_ood.')
parser.add_argument('--n_test_tokens_ood', nargs='+', required=False, default=[],
                    help='Number of unique tokens in OOD test sets. Default: n_test_tokens.')
parser.add_argument('--n_devdis', type=int, default=-1,
                    help='Total # developmental dissociation stimuli. Default: equal to n_train.')
parser.add_argument('--n_devdis_tokens', type=int, default=-1,
                    help='# unique developmental dissociation tokens. Default: equal to n_val_tokens.')
parser.add_argument('--generate_different_devdis', action='store_true', help='Generate different pairs \
                    for devdis sets.')

# Paremeters for logging, storing models, etc.
parser.add_argument('--save_model_freq', help='Number of times to save model checkpoints \
                    throughout training. Saves are equally spaced from 0 to num_epoch.', type=int,
                    default=-1)
parser.add_argument('--checkpoint', help='Whether or not to store model checkpoints.', action='store_true',
                    default=True)
parser.add_argument('--log_preds_freq', help='Number of times to log model predictions \
                    on test sets throughout training. Saves are equally spaced from 0 to num_epochs.',
                    type=int, default=-1)
parser.add_argument('--wandb_cache_dir', help='Directory for WandB cache. May need to be cleared \
                    depending on available storage in order to store artifacts.', default=None)
parser.add_argument('--clip_dir', help='Directory where CLIP models should be downloaded.',
                    default=None)
parser.add_argument('--wandb_run_dir', help='Directory where WandB runs should be stored.',
                    default=None)

args = parser.parse_args()

# Parse command line arguments
wandb_proj = args.wandb_proj
wandb_entity = args.wandb_entity

model_type = args.model_type
patch_size = args.patch_size
cnn_size = args.cnn_size
feature_extract = args.feature_extract
pretrained = args.pretrained

train_dataset_names = args.train_datasets
val_datasets_names = args.val_datasets
optim = args.optim
lr = args.lr
lr_scheduler = args.lr_scheduler
num_epochs = args.num_epochs
batch_size = args.batch_size
seed = args.seed

k = args.k
rotation = args.rotation
scaling = args.scaling
unaligned = False
args.unaligned = False
multiplier = args.multiplier

n_train = args.n_train
n_train_tokens = args.n_train_tokens
n_val = args.n_val
n_val_tokens = args.n_val_tokens
n_test = args.n_test
n_test_tokens = args.n_test_tokens
n_train_ood = args.n_train_ood
n_train_tokens_ood = args.n_train_tokens_ood
n_val_ood = args.n_val_ood
n_val_tokens_ood = args.n_val_tokens_ood
n_test_ood = args.n_test_ood
n_test_tokens_ood = args.n_test_tokens_ood
n_devdis = args.n_devdis
n_devdis_tokens = args.n_devdis_tokens
generate_different_devdis = args.generate_different_devdis

# make deterministic if given a seed 
if seed != -1:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# Default behavior for n_val, n_test
if val_datasets_names == 'all':
    val_datasets_names = [name for name in os.listdir('stimuli/source') if os.path.isdir(f'stimuli/source/{name}')]
    
    for td in train_dataset_names:
        val_datasets_names.remove(td)
        
for vd in val_datasets_names:
    if 'COMPLEX' in vd:
        val_datasets_names.remove(vd)

val_datasets_names = train_dataset_names

if n_val == -1:
    n_val = n_train
if n_test == -1:
    n_test = n_train
if n_devdis == -1:
    n_devdis = n_val
    
if len(n_train_ood) == 0:
    n_train_ood = [n_train for _ in range(len(val_datasets_names))]
elif len(n_train_ood) == 1:
    n_train_ood = [int(n_train_ood[0]) for _ in range(len(val_datasets_names))]
if len(n_val_ood) == 0:
    n_val_ood = copy.deepcopy(n_train_ood)
elif len(n_val_ood) == 1:
    n_val_ood = [int(n_val_ood[0]) for _ in range(len(val_datasets_names))]
if len(n_test_ood) == 0:
    n_test_ood = copy.deepcopy(n_train_ood)
elif len(n_test_ood) == 1:
    n_test_ood = [int(n_test_ood[0]) for _ in range(len(val_datasets_names))]

# Ensure that stimuli are 64x64
'''
if model_type == 'vit' or model_type == 'clip_vit':
    if patch_size == 16:
        multiplier = multiplier*2
'''

# Other hyperparameters/variables
im_size = 224
decay_rate = 0.95  # scheduler decay rate for Exponential type
int_to_label = {0: 'different', 1: 'same'}
label_to_int = {'different': 0, 'same': 1}

# Check arguments
#assert not (model_type == 'clip_rn' and cnn_size != 50)  # Only CLIP ResNet-50 is defined
assert not (model_type == 'clip_rn')  # Removing support for ResNet
assert len(n_train_ood) == len(val_datasets_names)
assert len(n_test_ood) == len(val_datasets_names)
assert len(n_val_ood) == len(val_datasets_names)
assert im_size % patch_size == 0
assert k == 2 or k == 4 or k == 8
assert model_type == 'resnet' or model_type == 'vit' or model_type == 'clip_rn' \
    or model_type == 'clip_vit'

# Create necessary directories 
try:
    os.mkdir('logs')
except FileExistsError:
    pass

# Create strings for paths and directories 
if unaligned:
    pos_string = 'unaligned'
else:
    pos_string = 'aligned'
    
if pretrained:
    pretrained_string = '_pretrained'
else:
    pretrained_string = ''
    
if feature_extract:
    fe_string = '_fe'
    
    try:
        os.mkdir('features')
    except FileExistsError:
        pass
else:
    fe_string = ''
    
aug_string = ''
if rotation:
    aug_string += 'R'
if scaling:
    aug_string += 'S'
if len(aug_string) == 0:
    aug_string = 'N'
    
if not unaligned:
    aug_string += f'_{patch_size}'

# Load models
if model_type == 'resnet':
    model_string = 'resnet_{0}'.format(cnn_size)
    
    if cnn_size == 18:
        model = models.resnet18(pretrained=pretrained)
    elif cnn_size == 50:
        model = models.resnet50(pretrained=pretrained)
        ckpt = torch.load('resnet50_miil_21k.pth')
        model.load_state_dict(ckpt)
    elif cnn_size == 152:
        model = models.resnet152(pretrained=pretrained)

    # Freeze layers if feature_extract
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
elif model_type == 'vit':
    model_string = 'vit_b{0}'.format(patch_size)
    model_path = f'google/vit-base-patch{patch_size}-{im_size}-in21k'

    if pretrained:
        model = ViTForImageClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label=int_to_label,
            label2id=label_to_int
        )
    else:
        configuration = ViTConfig(patch_size=patch_size, image_size=im_size)
        model = ViTForImageClassification(configuration)
        
    transform = ViTImageProcessor(do_resize=False).from_pretrained(model_path)

    if feature_extract:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            
elif 'clip' in model_type:                
    if 'vit' in model_type:
        model_string = 'clip_vit_b{0}'.format(patch_size)
        model_path = f"openai/clip-vit-base-patch{patch_size}"
        
        if pretrained:
            #model, transform = clip.load(f'ViT-B/{patch_size}', device=device, download_root=args.clip_dir)
            model = CLIPVisionModelWithProjection.from_pretrained(
                model_path, 
                hidden_act="quick_gelu", 
                id2label=int_to_label,
                label2id=label_to_int)
        else:
            configuration = CLIPConfig(patch_size=patch_size, im_size=im_size)
            model = CLIPVisionModelWithProjection(configuration)
            
        transform = AutoProcessor.from_pretrained(model_path)
        
        #in_features = model.visual.proj.shape[1]
        # Replace projection with correct dimensions
        in_features = model.visual_projection.in_features
        model.visual_projection = nn.Linear(in_features, 2)
    else:
        sys.exit(1)  # removing support for ResNet
        '''
        model_string = 'clip_resnet50'
        
        if pretrained:
            model, transform = clip.load('RN50', device=device, download_root=args.clip_dir)
        else:
            sys.exit(1)
        in_features = model.visual.output_dim
        '''
    
    if feature_extract:
        for name, param in model.visual.named_parameters():
            param.requires_grad = False
    
    '''
    # Add classification head to vision encoder
    fc = nn.Linear(in_features, 2).to(device)
    model = nn.Sequential(model.visual, fc).float()
    '''

model = model.to(device)  # Move model to GPU if possible

if feature_extract:
    backbone = nn.Sequential(*list(model.children())[:-1])
    classifier = nn.Sequential(list(model.children())[-1])
    model = {'backbone': backbone, 'classifier': classifier}
else:
    model = {'classifier': model}

# Create paths
model_string += pretrained_string  # Indicate if model is pretrained
model_string += fe_string  # Add 'fe' if applicable
model_string += '_{0}'.format(optim)  # Optimizer string

if len(train_dataset_names) == 1:
    train_dataset_string = train_dataset_names[0]
else:
    train_dataset_string = ''
    
    for td in train_dataset_names:
        if 'GRAYSCALE' in td:
            train_dataset_string += f'GRAY_{td.split("_")[1][:3]}-'
        elif 'MASK' in td:
            train_dataset_string += f'MASK_{td.split("_")[1][:3]}-'
        else:
            train_dataset_string += f'{td[:3]}-'
    train_dataset_string = train_dataset_string[:-1]

# Compute number of unique train tokens
n_unique = len([f for f in os.listdir(f'stimuli/source/{train_dataset_string}/{patch_size}') 
                if os.path.isfile(os.path.join(f'stimuli/source/{train_dataset_string}/{patch_size}', f)) 
                and f != '.DS_Store'])
if n_train_tokens == -1:
    percent_train = n_train / (n_train + n_val + n_test)
    percent_val = n_val / (n_train + n_val + n_test)
    percent_test = n_test / (n_train + n_val + n_test)
    
    n_unique_train = floor(n_unique * percent_train)
    n_unique_val = floor(n_unique * percent_val)
    n_unique_test = floor(n_unique * percent_test)
else:
    if args.ood:
        assert n_train_tokens <= n_unique - 2
        n_unique_train = n_train_tokens
        
        remainder = n_unique - n_train_tokens
        if n_val_tokens == -1:
            if n_test_tokens == -1:
                n_unique_val = remainder // 2
                n_unique_test = remainder // 2
            else:
                assert n_test_tokens < remainder
                n_unique_val = remainder - n_test_tokens
                n_unique_test = n_test_tokens
        else:
            if n_test_tokens == -1:
                assert n_val_tokens < remainder
                n_unique_val = n_val_tokens
                n_unique_test = remainder - n_val_tokens
            else:
                assert n_val_tokens + n_test_tokens <= remainder
                n_unique_val = n_val_tokens
                n_unique_test = n_test_tokens
    else:
        assert n_val_tokens == n_train_tokens
        assert n_test_tokens == n_train_tokens
        n_unique_train = n_train_tokens
        n_unique_val = n_train_tokens
        n_unique_test = n_train_tokens
            
if n_devdis_tokens == -1:
    n_devdis_tokens = n_val_tokens

if args.ood:
    if len(n_train_tokens_ood) == 0:
        n_train_tokens_ood = [n_unique_train for _ in range(len(val_datasets_names))]
    elif len(n_train_tokens_ood) == 1:
        n_train_tokens_ood = [int(n_train_tokens_ood[0]) for _ in range(len(val_datasets_names))]
    else:
        assert len(n_train_tokens_ood) == len(val_datasets_names)
        
    if len(n_val_tokens_ood) == 0:
        n_val_tokens_ood = [n_unique_val for _ in range(len(val_datasets_names))]
    elif len(n_val_tokens_ood) == 1:
        n_val_tokens_ood = [int(n_val_tokens_ood[0]) for _ in range(len(val_datasets_names))]
    else:
        assert len(n_val_tokens_ood) == len(val_datasets_names)   
        
    if len(n_test_tokens_ood) == 0:
        n_test_tokens_ood = [n_unique_test for _ in range(len(val_datasets_names))]
    elif len(n_test_tokens_ood) == 1:
        n_test_tokens_ood = [int(n_test_tokens_ood[0]) for _ in range(len(val_datasets_names))]
    else:
        assert len(n_test_tokens_ood) == len(val_datasets_names) 

path_elements = [model_string, train_dataset_string, pos_string, aug_string, f'trainsize_{n_train}_{n_unique_train}-{n_unique_val}-{n_unique_test}']

for root in ['logs']:
    stub = root

    for p in path_elements:
        try:
            os.mkdir('{0}/{1}'.format(stub, p))
        except FileExistsError:
            pass
        stub = '{0}/{1}'.format(stub, p)

log_dir = 'logs/{0}/{1}/{2}/{3}/{4}'.format(model_string, train_dataset_string, pos_string, aug_string, 
                                            f'trainsize_{n_train}_{n_unique_train}-{n_unique_val}-{n_unique_test}')

# Construct train set + DataLoader
train_dir = 'stimuli/{0}/{1}/{2}/{3}'.format(train_dataset_string, pos_string, aug_string, 
                                             f'trainsize_{n_train}_{n_unique_train}-{n_unique_val}-{n_unique_test}')

if not os.path.exists(train_dir):
    print(f"generating {train_dir}")
    call_create_stimuli(patch_size, n_train, n_val, n_test, k, unaligned, multiplier, 
                        train_dir, rotation, scaling, n_train_tokens=n_train_tokens, 
                        n_val_tokens=n_val_tokens, n_test_tokens=n_test_tokens)
    
train_dataset = SameDifferentDataset(train_dir + '/train', transform=transform, rotation=rotation, scaling=scaling)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_gpus, drop_last=True)

val_dataset = SameDifferentDataset(train_dir + '/val', transform=transform, rotation=rotation, scaling=scaling)
val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=True)
    
# Construct other validation sets
val_datasets = [val_dataset]
val_dataloaders = [val_dataloader]
val_labels = [train_dataset_string]

# Optimizer and scheduler
if optim == 'adamw':
    optimizer = torch.optim.AdamW(model['classifier'].parameters(), lr=lr)
elif optim == 'adam':
    optimizer = torch.optim.Adam(model['classifier'].parameters(), lr=lr)
elif optim == 'sgd':
    optimizer = torch.optim.SGD(model['classifier'].parameters(), lr=lr)

if lr_scheduler == 'reduce_on_plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=num_epochs//5)
elif lr_scheduler == 'exponential':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
elif lr_scheduler.lower() == 'none':
    scheduler = None

# Information to store
exp_config = {
    'model_type': model_type,
    'patch_size': patch_size,
    'cnn_size': cnn_size,
    'feature_extract': feature_extract,
    'pretrained': pretrained,
    'train_device': device,
    'train_dataset': train_dataset_string,
    'pos_condition': pos_string,
    'aug': aug_string,
    'train_size': n_train,
    'n_train_tokens': n_unique_train,
    'n_val_tokens': n_unique_val,
    'n_test_tokens': n_test_tokens,
    'learning_rate': lr,
    'scheduler': lr_scheduler,
    'decay_rate': decay_rate,
    'patience': num_epochs//5,
    'optimizer': optim,
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'stimulus_size': '{0}x{0}'.format(patch_size * multiplier),
}

# Initialize Weights & Biases project & table
if wandb_entity:
    run = wandb.init(project=wandb_proj, config=exp_config, entity=wandb_entity, dir=args.wandb_run_dir,
                     settings=wandb.Settings(start_method="fork"))
else:
    run = wandb.init(project=wandb_proj, config=exp_config, dir=args.wandb_run_dir,
                     settings=wandb.Settings(start_method="fork"))

run_id = wandb.run.id
run.name = f'TRAIN_{model_string}_{train_dataset_string}{n_train}-{n_unique_train}-{n_unique_val}-{n_unique_test}_{aug_string}_LR{lr}_{run_id}'

# Log model predictions
pred_columns = ['Training Epoch', 'File Name', 'Image', 'Dataset', 'Prediction',
                'Truth', 'Same Score', 'Different Score', 'Same Accuracy', 
                'Different Accuracy']
test_table = wandb.Table(columns=pred_columns)

# Run training loop + evaluations
model = train_model(args, model, device, train_dataloader, len(train_dataset), 
                    optimizer, scheduler, log_dir, val_datasets, val_dataloaders,
                    test_table, val_labels)
wandb.finish()
