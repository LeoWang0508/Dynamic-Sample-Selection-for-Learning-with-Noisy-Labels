from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import sys
import argparse
import numpy as np
from InceptionResNetV2 import *
from sklearn.mixture import GaussianMixture
import dataloader_webvision as dataloader
import torchnet
from Contrastive_loss import *

parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')   ## 0.01
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss') #0.025
parser.add_argument('--d_u',  default=0.7, type=float)
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--id', default='',type=str)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--data_path', default='./data/webvision/', type=str, help='path to dataset')
parser.add_argument('--resume', default=False , type=bool, help='Resume from chekpoint')
parser.add_argument('--dataset', default='WebVision', type=str)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
contrastive_criterion = SupConLoss()

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)

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

## Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net2.eval()         # Fix one network and train the other    
    net.train()       

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter_un  = (len(unlabeled_trainloader.dataset)//args.batch_size)+1
    num_iter_lab = (len(labeled_trainloader.dataset)//args.batch_size)+1

    loss_x = 0
    loss_u = 0
    loss_scl = 0
    loss_ucl = 0
    loss_ucl2 = 0

    num_iter = num_iter_lab
    
    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader):      
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
            
            with torch.no_grad():

                # Label co-guessing of unlabeled samples
                _, outputs_u11,_ = net(inputs_u3)
                _, outputs_u12,_ = net(inputs_u4)
                _, outputs_u21,_ = net2(inputs_u3)
                _, outputs_u22,_ = net2(inputs_u4)            
                
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
                ptu = pu**(1/args.T)                             ## Temparature Sharpening
                
                targets_u = ptu / ptu.sum(dim=1, keepdim=True)   ## Normalize
                targets_u = targets_u.detach()     

                ## Label refinement of labeled samples
                _, outputs_x,_  = net(inputs_x3)
                _, outputs_x2,_ = net(inputs_x4)            
                
                px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

                px = w_x*labels_x + (1-w_x)*px              
                ptx = px**(1/args.T)                            ## Temparature sharpening 
                            
                targets_x = ptx / ptx.sum(dim=1, keepdim=True)  ## normalize           
                targets_x = targets_x.detach()

            ## Mixmatch
            l = np.random.beta(args.alpha, args.alpha)        
            l = max(l,1-l)

            ## Unsupervised Contrastive Loss
            _,_,f1 = net(inputs_u)
            _,_,f2 = net(inputs_u2)
            f1    = F.normalize(f1, dim=1)
            f2    = F.normalize(f2, dim=1)
            features    = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_simCLR = contrastive_criterion(features)

            all_inputs  = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))
            input_a , input_b   = all_inputs , all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            
            ## Mixing inputs 
            mixed_input  = (l * input_a[: batch_size * 2] + (1 - l) * input_b[: batch_size * 2])
            mixed_target = (l * target_a[: batch_size * 2] + (1 - l) * target_b[: batch_size * 2])
                    
            _, logits,_ = net(mixed_input)

            Lx = -torch.mean(
                torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1)
            )

            ## Regularization
            prior = torch.ones(args.num_class) / args.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = Lx + args.lambda_c*loss_simCLR + penalty
            loss_x += Lx.item()
            loss_ucl += loss_simCLR.item()
            ## Compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('\r')
            sys.stdout.write('%s: | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Contrative Loss:%.4f'
                    %(args.dataset,  epoch, args.num_epochs, batch_idx+1, num_iter, loss_x/(batch_idx+1), loss_ucl/(batch_idx+1)))
            sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)[1]
        
        loss = CEloss(outputs, labels)   

        L = loss     

        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t CE-loss: %.4f'
                %(args.id, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
        
        
def test(epoch,net1,net2,test_loader):
    acc_meter.reset()

    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)[1]
            outputs2 = net2(inputs)[1]           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)     

            acc_meter.add(outputs,targets)

    accs = acc_meter.value()

    return accs

# Calculate the KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## JSD Calculation
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

def Calculate_JSD1(model1, model2,list_out1):   
    model1.eval()
    model2.eval() 
    num_iter = (len(eval_loader.dataset)//eval_loader.batch_size)+1
    JSD = torch.zeros(len(eval_loader.dataset))  
    JSD2 = torch.zeros(len(eval_loader.dataset))  
    JS_dist = Jensen_Shannon()

    for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])     
            out2 = torch.nn.Softmax(dim=1).cuda()(model2(inputs)[1])

        ## Get the Prediction
        out = (out1 + out2)/2     

        list_out1.append(out1.detach())

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        dist = JS_dist(out,  F.one_hot(targets, num_classes = args.num_class))
        

        if len(list_out1) > 690:
            dist2 = JS_dist(out1, list_out1[batch_idx]) + JS_dist(out1, list_out1[batch_idx+138]) + JS_dist(out1, list_out1[batch_idx+276]) + JS_dist(out1, list_out1[batch_idx+414]) + JS_dist(out1, list_out1[batch_idx+552]) 


        if batch_idx != num_iter-1:
            JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist
            if len(list_out1) > 690:
                JSD2[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist2
        else:
            JSD[int(batch_idx*eval_loader.batch_size):len(JSD)] = dist
            if len(list_out1) > 690:
                JSD2[int(batch_idx*eval_loader.batch_size):len(JSD)] = dist2

    print(len(list_out1))
    
    if len(list_out1) > 690:
        del list_out1[0:138]

    print(JSD)
    print(JSD2)  

    return JSD2, JSD

def Calculate_JSD2(model1, model2,list_out2):    
    model1.eval()
    model2.eval()
    num_iter = (len(eval_loader.dataset)//eval_loader.batch_size)+1
    JSD = torch.zeros(len(eval_loader.dataset))  
    JSD2 = torch.zeros(len(eval_loader.dataset))  
    JS_dist = Jensen_Shannon()

    for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
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

        if len(list_out2) > 690:
            dist2 = JS_dist(out2, list_out2[batch_idx]) + JS_dist(out2, list_out2[batch_idx+138]) + JS_dist(out2, list_out2[batch_idx+276]) + JS_dist(out2, list_out2[batch_idx+414]) + JS_dist(out2, list_out2[batch_idx+552]) 



        if batch_idx != num_iter-1:
            JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist
            if len(list_out2) > 690:
                JSD2[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist2
        else:
            JSD[int(batch_idx*eval_loader.batch_size):len(JSD)] = dist
            if len(list_out2) > 690:
                JSD2[int(batch_idx*eval_loader.batch_size):len(JSD)] = dist2

    print(len(list_out2))
    
    if len(list_out2) > 690:
        del list_out2[0:138]
        

    print(JSD)
    print(JSD2)      

    return JSD2, JSD


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
    model = InceptionResNetV2(num_classes=args.num_class)
    #model = model.cuda()
    return model

stats_log=open('./checkpoint/%s'%(args.id)+'_stats.txt','w') 
test_log=open('./checkpoint/%s'%(args.id)+'_acc.txt','w')     

warm_up = 2  
#mid_warmup = 25
loader = dataloader.webvision_dataloader(batch_size=args.batch_size,num_workers=6,root_dir=args.data_path,log=stats_log, num_class=args.num_class)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) 
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

folder = 'Webvision_Unicon'
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)

model_name_1 = 'Net1_120_2gpu_small2_32_0025.pth'
model_name_2 = 'Net2_120_2gpu_small2_32_0025.pth'
best_acc = 0
 

if args.resume:
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1))['net'])
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_2))['net'])
    start_epoch = warm_up
else:
    start_epoch = 0

device_ids = [0,1]
net1 = torch.nn.DataParallel(net1, device_ids=device_ids)
net2 = torch.nn.DataParallel(net2, device_ids=device_ids)
net1 = net1.to(device)
net2 = net2.to(device)


acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)


SR = 0
list_out1 = []
list_out2 = []


acc = 0
count = 0
count_label = 0
count_label2 = 0
prob1 = []
prob2 = []
new_targets = []
noisy_idx = []
pred_idx = []


for epoch in range(start_epoch,args.num_epochs+1):   

    # Manually Changing the learning rate ###
    lr=args.lr
    if epoch >= 60:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr  

    
    web_valloader = loader.run(0.5,0, 'test')
    imagenet_valloader = loader.run(0.5,0, 'imagenet') 
    eval_loader = loader.run(0.5,0, 'eval_train')  
    num_samples = len(eval_loader.dataset)
    print("Total Number of Samples: ", num_samples)

    if epoch<warm_up:     
        warmup_trainloader = loader.run(0.5,0, 'warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    ## Jump-Restart
    #elif (epoch+1)%mid_warmup==0:
    #    lr = 0.001                      #0.001
    #    for param_group in optimizer1.param_groups:
    #        param_group['lr'] = lr       
    #    for param_group in optimizer2.param_groups:
    #        param_group['lr'] = lr 

    #    warmup_trainloader = loader.run(0.5,0, 'warmup')
    #    print('Mid-training Warmup Net1')
    #    warmup(epoch,net1,optimizer1,warmup_trainloader)    
    #    print('\nMid-training Warmup Net2')
    #    warmup(epoch,net2,optimizer2,warmup_trainloader) 

    else:              
        prob1, prob = Calculate_JSD1(net1, net2,list_out1)

        threshold = torch.mean(prob)
        threshold1 = torch.mean(prob1)

        if threshold.item()>args.d_u:
            threshold = threshold - (threshold-torch.min(prob))/arg.tau

        if threshold1.item()>args.d_u:
            threshold1 = threshold1 - (threshold1-torch.min(prob1))/args.tau

        print(threshold)
        print(threshold1)
        
        if epoch <= warm_up+4:
            SR = torch.sum(prob<threshold).item()/num_samples
            print(SR)

        count = 0
        count2 = 0

        if epoch > warm_up+4:
            for i in range (num_samples):
                if prob[i] < threshold:  
                    if prob1[i] >= threshold1+threshold1*(1-threshold)*(1-(epoch-warm_up)/120) and prob1[i] >= threshold1:
                        prob[i] = 0.9
                        count += 1
                    
            print(count)
            print(count2)
                
            SR = torch.sum(prob<threshold).item()/num_samples
            print(SR)


        print('Train Net1')
        if epoch > warm_up+4:
            labeled_trainloader = loader.run(SR, 0, 'train', prob= prob) # Uniform Selection
        else:
            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 1, 'train', prob= prob)  # Uniform Selection
        
        if epoch > warm_up+4:
            feature_bank=[]
            feature_labels=[]
            noisy_idx = []
            list_feature = []
            list_feature_u = []
            list_labels = []
            count_label = 0
            count_label2 = 0

            with torch.no_grad():
                for batch_idx, (inputs_x, _, _, _, labels_x,_) in enumerate(labeled_trainloader):
                    inputs_x, labels_x = inputs_x.cuda(), labels_x.cuda()
                    
                ## Get outputs of both network
                
                    feature1 = torch.nn.Softmax(dim=1).cuda()(net1(inputs_x)[0]) 
                    
                    list_feature.append(feature1.detach())
                    list_labels.append(labels_x.detach())

            feature_bank = torch.cat(list_feature, dim=0).t().contiguous() 

            feature_labels = torch.cat(list_labels, dim=0).t().contiguous()
            knn_k = int(len(feature_labels)**0.5*0.9)
            

            with torch.no_grad():
                for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    
                    batch_size = inputs.size()[0]

                    ## Get outputs of both network
                    feature_u = torch.nn.Softmax(dim=1).cuda()(net1(inputs)[0]) 

                    pred_scores, pred_labels = knn_predict(feature_u, feature_bank, feature_labels, 50, knn_k)

                    for i in range(batch_size):
                        if prob[index[i]] > threshold:
                            if pred_labels[i] == targets[i]:
                                prob[index[i]] = 0.001
                                count_label += 1
                        else:
                            if pred_labels[i] != targets[i]:
                                if prob1[index[i]] > threshold1*(1-(epoch-warm_up)/120)+threshold1*(1-threshold)*(1-(epoch-warm_up)/120):
                                    prob[index[i]] = 0.9
                                    count_label2 += 1

            print(count_label)
            print(count_label2)

            SR = torch.sum(prob<threshold).item()/num_samples
            print(SR)

            labeled_trainloader, unlabeled_trainloader = loader.run(SR, epoch, 'train', prob= prob)

        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader)      

#########Net2##################################################################################################################################

        prob1, prob = Calculate_JSD2(net1,net2,list_out2)

        threshold = torch.mean(prob)
        threshold1 = torch.mean(prob1)

        if threshold.item()>args.d_u:
            threshold = threshold - (threshold-torch.min(prob))/arg.tau

        if threshold1.item()>args.d_u:
            threshold1 = threshold1 - (threshold1-torch.min(prob1))/args.tau

        print(threshold)
        print(threshold1)

        if epoch <= warm_up+4:
            SR = torch.sum(prob<threshold).item()/num_samples
            print(SR)

        count = 0
        count2 = 0

        if epoch > warm_up+4:
            for i in range (num_samples):
                if prob[i] < threshold:                
                    if prob1[i] >= threshold1+threshold1*(1-threshold)*(1-(epoch-warm_up)/120) and prob1[i] >= threshold1:
                        prob[i] = 0.9
                        count += 1
                    
            print(count)
            print(count2)
                
            SR = torch.sum(prob<threshold).item()/num_samples
            print(SR)


        print('Train Net2')
        if epoch > warm_up+4:
            labeled_trainloader = loader.run(SR, 0, 'train', prob= prob) # Uniform Selection
        else:
            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 1,'train', prob= prob)  # Uniform Selection

        if epoch > warm_up+4:
            feature_bank=[]
            feature_labels=[]
            noisy_idx = []
            list_feature = []
            list_feature_u = []
            list_labels = []
            count_label = 0
            count_label2 = 0
            count_label3 = 0

            with torch.no_grad():
                for batch_idx, (inputs_x, _, _, _, labels_x,_) in enumerate(labeled_trainloader):
                    inputs_x, labels_x = inputs_x.cuda(), labels_x.cuda()
                    
                ## Get outputs of both network
                
                    feature1 = torch.nn.Softmax(dim=1).cuda()(net2(inputs_x)[0]) 
                    

                    list_feature.append(feature1.detach())
                    list_labels.append(labels_x.detach())


            feature_bank = torch.cat(list_feature, dim=0).t().contiguous()
            feature_labels = torch.cat(list_labels, dim=0).t().contiguous()
            knn_k = int(len(feature_labels)**0.5*0.9)

            with torch.no_grad():
                for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()#targets.to(device2)
                    
                    batch_size = inputs.size()[0]


                    ## Get outputs of both network
                    feature_u = torch.nn.Softmax(dim=1).cuda()(net2(inputs)[0])      

                    pred_scores, pred_labels = knn_predict(feature_u, feature_bank, feature_labels, 50, knn_k)

                    for i in range(batch_size):
                        if prob[index[i]] > threshold:
                            if pred_labels[i] == targets[i]:
                                prob[index[i]] = 0.001
                                count_label += 1
                        else:
                            if pred_labels[i] != targets[i]:
                                if prob1[index[i]] > threshold1*(1-(epoch-warm_up)/120)+threshold1*(1-threshold)*(1-(epoch-warm_up)/120):
                                    prob[index[i]] = 0.9
                                    count_label2 += 1

            print(count_label)
            print(count_label2)

            SR = torch.sum(prob<threshold).item()/num_samples
            print(SR)

            labeled_trainloader, unlabeled_trainloader = loader.run(SR, epoch, 'train', prob= prob)
            
        train(epoch, net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader)       
   
    
    web_acc = test(epoch,net1,net2,web_valloader)  
    imagenet_acc = test(epoch,net1,net2,imagenet_valloader)  
    
    print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
    test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n'%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
    test_log.flush()  


    if web_acc[0] > best_acc:
        if epoch <warm_up:
            model_name_1 = 'Net1_warmup_120_2gpu_small2_32_0025.pth'
            model_name_2 = 'Net2_warmup_120_2gpu_small2_32_0025.pth'
        else:
            model_name_1 = 'Net1_120_2gpu_small2_32_0025.pth'
            model_name_2 = 'Net2_120_2gpu_small2_32_0025.pth'            

        print("Save the Model-----")
        checkpoint1 = {
            'net': net1.module.state_dict(),
            'optimizer_state_dict': optimizer1.state_dict(),
            'Model_number': 1,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Accuracy': web_acc,
            'Dataset': 'WebVision',
            'epoch': epoch,
        }

        checkpoint2 = {
            'net': net2.module.state_dict(),
            'optimizer_state_dict': optimizer2.state_dict(),
            'Model_number': 2,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Accuracy': web_acc,
            'Dataset': 'WebVision',
            'epoch': epoch,
        }

        torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
        torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))

        torch.save(list_out1,'./checkpoint/webvision2_list_out1_32_0025.pth')  
        torch.save(list_out2,'./checkpoint/webvision2_list_out2_32_0025.pth')  

        best_acc = web_acc[0]



