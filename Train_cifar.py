from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import random
import os
import argparse
import numpy as np
from PreResNet_cifar import *
import dataloader_cifar as dataloader
from math import log2
from Contrastive_loss import *
import collections.abc
from collections.abc import MutableMapping

from models.resnet import SupCEResNet
from sklearn.mixture import GaussianMixture
## For plotting the logs
# import wandb
# wandb.init(project="noisy-label-project", entity="..")

## Arguments to pass 
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate') #0.02
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=30, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.05, type=float, help='weight for contrastive loss') #0.025
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=350, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--d_u',  default=0.7, type=float)
parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
parser.add_argument('--metric', type=str, default = 'JSD', help='Comparison Metric')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool, help = 'Resume from the warmup checkpoint')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data/cifar10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']='0'
## GPU Setup 
torch.cuda.set_device(0)

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

## Download the Datasets
if args.dataset== 'cifar10':
    torchvision.datasets.CIFAR10(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR10(args.data_path,train=False, download=True)
else:
    torchvision.datasets.CIFAR100(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR100(args.data_path,train=False, download=True)

## Checkpoint Location
folder = args.dataset + '_' + args.noise_mode + '_' + str(args.r) 
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)

## Log files
stats_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     
test_loss_log = open(model_save_loc +'/test_loss.txt','w')
#train_acc = open(model_save_loc +'/train_acc.txt','w')
#train_loss = open(model_save_loc +'/train_loss.txt','w')

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    # print(sim_weight.shape, sim_labels.shape)
    sim_weight = torch.ones_like(sim_weight)

    sim_weight = sim_weight / sim_weight.sum(dim=-1, keepdim=True)

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argmax(dim=-1)
    return pred_scores, pred_labels

# SSL-Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader,warm_up):
    net2.eval() # Freeze one network and train the other
    net.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    labeled_train_iter = iter(labeled_trainloader)   
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    num_iter2 = (len(unlabeled_trainloader.dataset)//(args.batch_size*2))+1
    ## Loss statistics
    loss_x = 0
    loss_u = 0
    loss_scl = 0
    loss_ucl = 0
    loss_ucl2 = 0


    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, gt_label,w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)
                
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)
                
        batch_size = inputs_x.size(0)


        if batch_size>3:
            # Transform label to one-hot

            labels_x = torch.zeros(batch_size, args.num_class, device=inputs_x.device).scatter_(1, labels_x.view(-1,1), 1) 

            w_x = w_x.view(-1,1).type(torch.FloatTensor) 

            inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()

            batch_size_u = inputs_u.size(0)

            with torch.no_grad():
                # Label co-guessing of unlabeled samples
                _,outputs_u11,_  = net(inputs_u)
                _,outputs_u12,_  = net(inputs_u2)
                _,outputs_u21,_  = net2(inputs_u)
                _,outputs_u22,_  = net2(inputs_u2)   
         
                ## Pseudo-label
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4 
                    
                ptu = pu**(1/args.T)            ## Temparature Sharpening
                    
                targets_u = ptu / ptu.sum(dim=1, keepdim=True)
                    
                targets_u = targets_u.detach()   

                ## Label refinement
                _,outputs_x,_  = net(inputs_x)
                _,outputs_x2,_ = net(inputs_x2)  

                px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

                px = w_x*labels_x + (1-w_x)*px              
                ptx = px**(1/args.T)    ## Temparature sharpening

                targets_x = ptx / ptx.sum(dim=1, keepdim=True)           
                targets_x = targets_x.detach()

            ## Unsupervised Contrastive Loss
            
            f1, _, _ = net(inputs_u3)
            f2, _, _ = net(inputs_u4)
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_simCLR = contrastive_criterion(features)

            f1, _, _ = net(inputs_x3)
            f2, _, _ = net(inputs_x4)
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_simCLR2 = contrastive_criterion(features)

            # MixMatch
            l = np.random.beta(args.alpha, args.alpha)        
            l = max(l, 1-l)
                       
            all_inputs  = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))
                  
            input_a, input_b   = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
                
            ## Mixup
            mixed_input  = l * input_a  + (1 - l) * input_b        
            mixed_target = l * target_a + (1 - l) * target_b
                
            _,logits, _ = net(mixed_input)

            logits_x = logits[:batch_size*2]
            logits_u = logits[batch_size*2:] 

                
            ## Combined Loss
            Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
                
            ## Regularization
            prior = torch.ones(args.num_class)/args.num_class
            prior = prior.cuda()        
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior*torch.log(prior/pred_mean))

            ## Total Loss
            loss = Lx + lamb * Lu + args.lambda_c*loss_simCLR + args.lambda_c*loss_simCLR2 + penalty 

            ## Accumulate Loss
            loss_x += Lx.item()
            loss_u += Lu.item()
            loss_ucl += loss_simCLR.item()
            loss_ucl2 += loss_simCLR2.item()
            # Compute gradient and Do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.4f Contrastive Loss:%.4f Contrastive Loss2:%.4f'
                        %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss_x/(batch_idx+1), loss_u/(batch_idx+1), loss_ucl/(batch_idx+1),loss_ucl2/(batch_idx+1)))
            sys.stdout.flush()
            

## For Standard Training 
def warmup_standard(epoch,net,optimizer,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    
    for batch_idx, (inputs, labels, path,_) in enumerate(dataloader): 
   
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _, outputs, _ = net(inputs)               
        loss    = CEloss(outputs, labels)    

        if args.noise_mode=='asym':     # Penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        else:   
            L = loss

        L.backward()  
        optimizer.step()                

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

## For Training Accuracy
def warmup_val(epoch,net,optimizer,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    total = 0
    correct = 0
    loss_x = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels, path,_) in enumerate(dataloader):     

            inputs, labels = inputs.cuda(), labels.cuda() 
            optimizer.zero_grad()
            _, outputs, _  = net(inputs)               
            _, predicted = torch.max(outputs, 1)    
            loss    = CEloss(outputs, labels)    
            loss_x += loss.item()                      

            total   += labels.size(0)
            correct += predicted.eq(labels).cpu().sum().item()

    acc = 100.*correct/total
    print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch, acc))  
    
    train_loss.write(str(loss_x/(batch_idx+1)))
    train_acc.write(str(acc))
    train_acc.flush()
    train_loss.flush()

    return acc

## Test Accuracy
def test(epoch,net1,net2):
    net1.eval()
    net2.eval()

    num_samples = 1000
    correct = 0
    total = 0
    loss_x = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1, _ = net1(inputs)
            _, outputs2, _ = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
            loss = CEloss(outputs, targets)  
            loss_x += loss.item()

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()  

    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write(str(acc)+'\n')
    test_log.flush()  
    test_loss_log.write(str(loss_x/(batch_idx+1))+'\n')
    test_loss_log.flush()
    return acc

# KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

## Calculate JSD
def Calculate_JSD2_1(model1, model2, num_samples, list_out1):  
    JS_dist = Jensen_Shannon()
    JSD = np.zeros(num_samples)    
    JSD1 = np.zeros(num_samples)
    for batch_idx, (inputs, targets, _,_) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])     
            out2 = torch.nn.Softmax(dim=1).cuda()(model2(inputs)[1]) 
            ## Get the Prediction
            out = (out1 + out2)/2     

            list_out1.append(out1.detach())
            #new_targets.append(out.detach().argmax(dim=-1).cpu())
            ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
            dist = JS_dist(out, F.one_hot(targets, num_classes = args.num_class))
            JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist.cpu()
            if len(list_out1) > 2500:
                dist1 = JS_dist(out1, list_out1[batch_idx]) + JS_dist(out1, list_out1[batch_idx+500]) + JS_dist(out1, list_out1[batch_idx+1000]) + JS_dist(out1, list_out1[batch_idx+1500]) + JS_dist(out1, list_out1[batch_idx+2000])+JS_dist(out2, out1)+JS_dist(out2, out1)
                JSD1[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist1.cpu()
            
    if len(list_out1) > 2500:
        del list_out1[0:500]
        
    print(JSD)
    print(JSD1)

    return JSD1, JSD


def Calculate_JSD2_2(model1, model2, num_samples, list_out2):  
    JS_dist = Jensen_Shannon()
    JSD = np.zeros(num_samples)    
    JSD2 = np.zeros(num_samples)
    for batch_idx, (inputs, targets, _,_) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])     
            out2 = torch.nn.Softmax(dim=1).cuda()(model2(inputs)[1])
 
            ## Get the Prediction
            out = (out1 + out2)/2     

            list_out2.append(out2.detach())
            ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
            dist = JS_dist(out,  F.one_hot(targets, num_classes = args.num_class))
            JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist.cpu()
            if len(list_out2) > 2500:
                dist2 = JS_dist(out2, list_out2[batch_idx]) + JS_dist(out2, list_out2[batch_idx+500]) + JS_dist(out2, list_out2[batch_idx+1000]) + JS_dist(out2, list_out2[batch_idx+1500]) + JS_dist(out2, list_out2[batch_idx+2000]) 
                
                JSD2[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist2.cpu()

    if len(list_out2) > 2500:
        del list_out2[0:500]
        
    print("\n")
    print(JSD)
    print(JSD2)
    return JSD2, JSD

## Unsupervised Loss coefficient adjustment 
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model


## Choose Warmup period based on Dataset
num_samples = 50000
if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30


## Call the dataloader
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=4,\
    root_dir=model_save_loc,log=stats_log, noise_file='%s/clean_%.4f_%s.npz'%(args.data_path,args.r, args.noise_mode),gt_file='%s/clean_%.4f_%s_gt.npz'%(args.data_path,args.r, args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

## Semi-Supervised Loss
criterion  = SemiLoss()

## Optimizer and Scheduler
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 300, 2e-4)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, 300, 2e-4)

## Loss Functions
CE       = nn.CrossEntropyLoss(reduction='none')
CEloss   = nn.CrossEntropyLoss()
MSE_loss = nn.MSELoss(reduction= 'none')
contrastive_criterion = SupConLoss()

if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

## Resume from the warmup checkpoint 
model_name_1 = 'Net1_warmup.pth'
model_name_2 = 'Net2_warmup.pth'    

if args.resume:
    start_epoch = warm_up
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1))['net'])
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_2))['net'])

else:
    start_epoch = 0

best_acc = 0
test_loader = loader.run(0, 0, 0, 'test')
eval_loader = loader.run(0, 0, 0, 'eval_train')   
warmup_trainloader = loader.run(0, 0, 0, 'warmup')
acc = 0
list_out1 = []
list_out2 = []
count = 0
prob1 = []
prob2 = []

save_file = 'checkpoint/'+str(args.dataset)+'_'+str(args.noise_mode)+'_'+str(args.r)+'/Clean_index_'+ str(args.dataset) + '_' +str(args.noise_mode) +'_' + str(args.r) + '.npz'

## Warmup and SSL-Training 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for epoch in range(start_epoch,args.num_epochs+1):  

    ## Warmup Stage 
    if epoch<warm_up:    

        warmup_trainloader = loader.run(0, 0, epoch, 'warmup')

        print('\nWarmup Model1')
        warmup_standard(epoch, net1, optimizer1, warmup_trainloader) 

        print('\nWarmup Model2')
        warmup_standard(epoch, net2, optimizer2, warmup_trainloader) 
        
    else:
        ## Calculate JSD values and Filter Rate
        prob1, prob = Calculate_JSD2_1(net1, net2, num_samples, list_out1)

        threshold = np.mean(prob)
        threshold1 = np.mean(prob1)
        if threshold.item()>args.d_u:
            threshold = threshold - (threshold-np.min(prob))/args.tau

        if threshold1.item()>args.d_u:
            threshold1 = threshold1 - (threshold1-np.min(prob1))/args.tau

        print(threshold)
        print(threshold1)
        if epoch <= warm_up+4:
            SR = np.sum(prob<threshold).item()/num_samples
            print(SR)

        if epoch > warm_up+4:
            count = 0
            for i in range (num_samples):
                if prob[i] < threshold: 
                    if prob1[i] >= threshold1+threshold1*(1-threshold)*(1-(epoch-warm_up)/300) and prob1[i] >= threshold1:#+threshold1*(1-threshold):
                        prob[i] = 0.9
                        count += 1
                    
            print(count)

            SR = np.sum(prob<threshold).item()/num_samples
            print(SR)
        
        print('\nTrain Net1')
        if epoch > warm_up+4:
            labeled_trainloader = loader.run(SR, 0, 0, 'train', prob= prob) # Uniform Selection

        else:
            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 0, epoch, 'train', prob= prob)
        
        if epoch > warm_up+4:
            feature_bank=[]
            feature_labels=[]
            list_feature = []
            list_feature_u = []
            list_labels = []
            count_label = 0
            count_label2 = 0

            with torch.no_grad():
                for batch_idx, (inputs_x, _, _, _, labels_x,_,_) in enumerate(labeled_trainloader):
                    inputs_x, labels_x = inputs_x.cuda(), labels_x.cuda()

                ## Get outputs of both network
                
                    feature1 = torch.nn.Softmax(dim=1).cuda()(net1(inputs_x)[2]) 

                    list_feature.append(feature1.detach())
                    list_labels.append(labels_x.detach())


            feature_bank = torch.cat(list_feature, dim=0).t().contiguous()
            feature_labels = torch.cat(list_labels, dim=0).t().contiguous()
            knn_k = int(len(feature_labels)**0.5*0.9)
            
            with torch.no_grad():
                for batch_idx, (inputs, targets, index, gt_label) in enumerate(eval_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    batch_size = inputs.size()[0]

                    ## Get outputs of both network
                    feature_u = torch.nn.Softmax(dim=1).cuda()(net1(inputs)[2])      
                    
                    pred_scores, pred_labels = knn_predict(feature_u, feature_bank, feature_labels, args.num_class, knn_k)
                    for i in range(batch_size):
                        if prob[index[i]] > threshold:
                            if pred_labels[i] == targets[i]:
                                prob[index[i]] = 0.001
                                count_label += 1

                        else:
                            if pred_labels[i] != targets[i]:
                                if prob1[index[i]] > threshold1*(1-(epoch-warm_up)/300)+threshold1*(1-threshold)*(1-(epoch-warm_up)/300):
                                    prob[index[i]] = 0.9
                                    count_label2 += 1
                                            


            print(count_label)
            print(count_label2)


            SR = np.sum(prob<threshold).item()/num_samples
            print(SR)

            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 0, epoch, 'train', prob= prob)
        
        train(epoch,net1,net2,optimizer1,labeled_trainloader,unlabeled_trainloader,warm_up)
        
        ## Calculate JSD values and Filter Rate

        prob1, prob = Calculate_JSD2_2(net1, net2, num_samples, list_out2) 

        threshold = np.mean(prob)
        threshold1 = np.mean(prob1)
        if threshold.item()>args.d_u:
            threshold = threshold - (threshold-np.min(prob))/args.tau
        if threshold1.item()>args.d_u:
            threshold1 = threshold1 - (threshold1-np.min(prob1))/args.tau

        print(threshold)
        print(threshold1)
        if epoch <= warm_up+4:
            SR = np.sum(prob<threshold).item()/num_samples
            print(SR)

        if epoch > warm_up+4:
            count = 0

            for i in range (num_samples):
                if prob[i] < threshold:
                    if prob1[i] >= threshold1+threshold1*(1-threshold)*(1-(epoch-warm_up)/300) and prob1[i] >= threshold1:
                        prob[i] = 0.9
                        count += 1
                    
            print(count)


            SR = np.sum(prob<threshold).item()/num_samples
            print(SR)

        print('\nTrain Net2')
        
        if epoch > warm_up+4:
            labeled_trainloader = loader.run(SR, 0, 0, 'train', prob= prob) # Uniform Selection
        else:
            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 0, epoch, 'train', prob= prob)

        if epoch > warm_up+4:
            feature_bank=[]
            feature_labels=[]
            list_feature = []
            list_feature_u = []
            list_labels = []
            count_label = 0
            count_label2 = 0

            with torch.no_grad():
                for batch_idx, (inputs_x, _, _, _, labels_x,_, _) in enumerate(labeled_trainloader):
                    inputs_x, labels_x = inputs_x.cuda(), labels_x.cuda()
                
                ## Get outputs of both network
                
                    feature1 = torch.nn.Softmax(dim=1).cuda()(net2(inputs_x)[2]) 

                    list_feature.append(feature1.detach())
                    list_labels.append(labels_x.detach())

            feature_bank = torch.cat(list_feature, dim=0).t().contiguous()
            feature_labels = torch.cat(list_labels, dim=0).t().contiguous()
            knn_k = int(len(feature_labels)**0.5*0.9)

            with torch.no_grad():

                for batch_idx, (inputs, targets, index, gt_label) in enumerate(eval_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    batch_size = inputs.size()[0]

                    ## Get outputs of both network
                    feature_u = torch.nn.Softmax(dim=1).cuda()(net2(inputs)[2]) 
                    
                    pred_scores, pred_labels = knn_predict(feature_u, feature_bank, feature_labels, args.num_class, knn_k)

                    for i in range(batch_size):
                        if prob[index[i]] > threshold:     
                            if pred_labels[i] == targets[i]:
                                prob[index[i]] = 0.001
                                count_label += 1
                            
                        else:
                            if pred_labels[i] != targets[i]:  
                                if prob1[index[i]] > threshold1*(1-(epoch-warm_up)/300)+threshold1*(1-threshold)*(1-(epoch-warm_up)/300):
                                    prob[index[i]] = 0.9                    
                                    count_label2 += 1
                                       
            print(count_label)
            print(count_label2)


            SR = np.sum(prob<threshold).item()/num_samples
            print(SR)
            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 0, epoch, 'train', prob= prob)

        train(epoch,net2,net1,optimizer2,labeled_trainloader,unlabeled_trainloader,warm_up)

    acc = test(epoch,net1,net2) 

    
    scheduler1.step()
    scheduler2.step()

    if epoch == warm_up-1:
        model_name_1 = 'Net1_warmup.pth'
        model_name_2 = 'Net2_warmup.pth'

        print("Save the Model-----")
        checkpoint1 = {
            'net': net1.state_dict(),
            'Model_number': 1,
            'Noise_Ratio': args.r,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Accuracy': acc,
            'Pytorch version': '1.4.0',
            'Dataset': 'cifar100',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        checkpoint2 = {
            'net': net2.state_dict(),
            'Model_number': 2,
            'Noise_Ratio': args.r,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Accuracy': acc,
            'Pytorch version': '1.4.0',
            'Dataset': 'cifar100',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
        torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))
        best_acc = acc

    elif epoch >= warm_up:
        if acc > best_acc:
            model_name_1 = 'Net1.pth'
            model_name_2 = 'Net2.pth'            

            print("Save the Model-----")
            checkpoint1 = {
                'net': net1.state_dict(),
                'Model_number': 1,
                'Noise_Ratio': args.r,
                'Loss Function': 'CrossEntropyLoss',
                'Optimizer': 'SGD',
                'Noise_mode': args.noise_mode,
                'Accuracy': acc,
                'Pytorch version': '1.4.0',
                'Dataset': 'cifar100',
                'Batch Size': args.batch_size,
                'epoch': epoch,
            }

            checkpoint2 = {
                'net': net2.state_dict(),
                'Model_number': 2,
                'Noise_Ratio': args.r,
                'Loss Function': 'CrossEntropyLoss',
                'Optimizer': 'SGD',
                'Noise_mode': args.noise_mode,
                'Accuracy': acc,
                'Pytorch version': '1.4.0',
                'Dataset': 'cifar100',
                'Batch Size': args.batch_size,
                'epoch': epoch,
            }

            torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
            torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))
            best_acc = acc

