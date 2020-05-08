import json
from tqdm import tqdm
import numpy as np
import os
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from ntm import NTM
from ntm.datasets import ReverseDataset
from ntm.args import get_parser
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------------------------
# Each time train the module:
# 1.usage : train.py [-h] [-task_json TASK_JSON] [-batch_size BATCH_SIZE]
#                    [-num_iters NUM_ITERS] [-config CONFIG] [-lr LR] [-momentum MOMENTUM]
#                    [-alpha ALPHA] [-saved_model SAVED_MODEL] [-beta1 BETA1] [-beta2 BETA2]
# 2.Try different configurations: python train.py -config m1d1
#           --Three configurations: m1 m2 m3
#           --Three data samples:   d1 d2 d3
# ----------------------------------------------------------------------------

# 1.tensorboard
args = get_parser().parse_args()
writer = SummaryWriter(log_dir='runs/'+args.config+'/train1')
writer2 = SummaryWriter(log_dir='runs/'+args.config+'/test2')
writer3 = SummaryWriter(log_dir='runs/'+args.config+'/test3')
loss_name = 'loss_'+args.config
# 2.NTM configuration
m=args.config[1]
args.task_json = 'ntm/tasks/reverse'+m+'.json'
task_params = json.load(open(args.task_json))
# 3.data configuration
d=int(args.config[3])-1
data = [[1,20,20,100], [1,10,10,100], [1,30,30,100]]
task_params['min_seq_len'] = data[d][0]
task_params['max_seq_len'] = data[d][1]
dataset = ReverseDataset(task_params)
dataset2 = ReverseDataset(task_params)
task_params['min_seq_len'] = data[d][2]
task_params['max_seq_len'] = data[d][3]
dataset3 = ReverseDataset(task_params)
#4.save model
args.saved_model = 'saved_model/'+'saved_model_reverse_'+args.config+'.pt'
cur_dir = os.getcwd()
PATH = os.path.join(cur_dir, args.saved_model)

"""
For the Reverse task, input_size: seq_width + 2, output_size: seq_width
"""
ntm = NTM(input_size=task_params['seq_width'] + 2,
          output_size=task_params['seq_width'],
          controller_size=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'])

criterion = nn.BCELoss()
# As the learning rate is task specific, the argument can be moved to json file
# optimizer = optim.RMSprop(ntm.parameters(),
#                           lr=args.lr,
#                           alpha=args.alpha,
#                           momentum=args.momentum)
optimizer = optim.Adam(ntm.parameters(), lr=args.lr,
                       betas=(args.beta1, args.beta2))

# ----------------------------------------------------------------------------
# -- basic training loop
# ----------------------------------------------------------------------------
losses = []
losses2 = []
losses3 = []
errors = []
for iter in tqdm(range(args.num_iters)):
    optimizer.zero_grad()
    ntm.reset()

    data = dataset[iter]
    input, target = data['input'].to(device), data['target'].to(device)
    out = torch.zeros(target.size()).to(device)

    # -------------------------------------------------------------------------
    # loop for other tasks
    # -------------------------------------------------------------------------
    for i in range(input.size()[0]):
        # to maintain consistency in dimensions as torch.cat was throwing error
        # in_data = torch.unsqueeze(input[i], 0).cuda()
        in_data = torch.unsqueeze(input[i], 0)
        ntm(in_data)

    # passing zero vector as input while generating target sequence
    # in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0).cuda()
    in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0).to(device)
    for i in range(target.size()[0]):
        out[i] = ntm(in_data)

    loss = criterion(out, target)
    losses.append(loss.item())

    binary_output = out.clone()
    binary_output = binary_output.detach().cpu().apply_(lambda x: 0 if x < 0.5 else 1)

    # sequence prediction error is calculted in bits per sequence
    error = torch.sum(torch.abs(binary_output - target.detach().cpu()))
    errors.append(error.item())

    #calculating test2 loss and test3 loss
    with torch.no_grad():
        # ---logging---
        if iter % 100 == 0:
            # test1_same_length
            ntm.reset()
            data2 = dataset2[iter]
            input2, target2 = data2['input'].to(device), data2['target'].to(device)
            out2 = torch.zeros(target2.size()).to(device)
            for i in range(input2.size()[0]):
                in_data2 = torch.unsqueeze(input2[i], 0)
                ntm(in_data2)
            in_data2 = torch.unsqueeze(torch.zeros(input2.size()[1]), 0).to(device)
            for i in range(target2.size()[0]):
                out2[i] = ntm(in_data2)
            loss2 = criterion(out2, target2)
            losses2.append(loss2.item())
            
            # test2_longer_length
            ntm.reset()
            data3 = dataset3[iter]
            input3, target3 = data3['input'].to(device), data3['target'].to(device)
            out3 = torch.zeros(target3.size()).to(device)
            for i in range(input3.size()[0]):
                in_data3 = torch.unsqueeze(input3[i], 0)
                ntm(in_data3)
            in_data3 = torch.unsqueeze(torch.zeros(input3.size()[1]), 0).to(device)
            for i in range(target3.size()[0]):
                out3[i] = ntm(in_data3)
            loss3 = criterion(out3, target3)
            losses3.append(loss3.item())

            #print loss and tensorboard
            print('Iteration: %d\tLoss: %.4f\tLoss2: %.4f\tLoss3: %.4f\tError in bits per sequence: %.2f\tSize: %2d\tSize2: %2d\tSize3: %3d' %
                (iter, np.mean(losses), np.mean(losses2), np.mean(losses3), np.mean(errors), input.size()[0], input2.size()[0], input3.size()[0]))
            writer.add_scalar(loss_name, np.mean(losses), iter)
            writer2.add_scalar(loss_name, np.mean(losses2), iter)
            writer3.add_scalar(loss_name, np.mean(losses3), iter)
            
            losses = []
            losses2 = []
            losses3 = []
            errors = []
    loss.backward()
    # clips gradient in the range [-10,10]. Again there is a slight but
    # insignificant deviation from the paper where they are clipped to (-10,10)
    nn.utils.clip_grad_value_(ntm.parameters(), 10)
    optimizer.step()
# ---calculate total trainable parameters
total_trainable_params = sum(p.numel() for p in ntm.parameters() if p.requires_grad)
print('total_trainable_params', total_trainable_params)

# ---saving the model---
torch.save(ntm.state_dict(), PATH)
# torch.save(ntm, PATH)
writer.close()
writer2.close()
writer3.close()