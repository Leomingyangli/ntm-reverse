import json
import os
import torch
from torch import nn
from ntm import NTM
from ntm.datasets import ReverseDataset
from ntm.args import get_parser

device = "cuda:0" if torch.cuda.is_available() else "cpu"
args = get_parser().parse_args()

m=args.config[1]
args.task_json = 'ntm/tasks/reverse'+m+'.json'

task_params = json.load(open(args.task_json))
criterion = nn.BCELoss()

# ---Evaluation parameters for Reverse task---
d=int(args.config[3])-1
data = [[1,20,20,100], [1,15,15,100], [1,20,20,300]]
task_params['min_seq_len'] = data[d][2]
task_params['max_seq_len'] = data[d][3]
dataset = ReverseDataset(task_params)
args.saved_model = 'saved_model/'+'saved_model_reverse_'+args.config+'.pt'

cur_dir = os.getcwd()
PATH = os.path.join(cur_dir, args.saved_model)

ntm = NTM(input_size=task_params['seq_width'] + 2,
          output_size=task_params['seq_width'],
          controller_size=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'])

ntm.load_state_dict(torch.load(PATH))

# -----------------------------------------------------------------------------
# --- evaluation
# -----------------------------------------------------------------------------
ntm.reset()
data = dataset[0]  # 0 is a dummy index
_input, target = data['input'].to(device), data['target'].to(device)
out = torch.zeros(target.size()).to(device)

# -----------------------------------------------------------------------------
# loop for other tasks
# -----------------------------------------------------------------------------
for i in range(_input.size()[0]):
    # to maintain consistency in dimensions as torch.cat was throwing error
    in_data = torch.unsqueeze(_input[i], 0)
    ntm(in_data)

# passing zero vector as the _input while generating target sequence
in_data = torch.unsqueeze(torch.zeros(_input.size()[1]), 0).to(device)
for i in range(target.size()[0]):
    out[i] = ntm(in_data)

loss = criterion(out, target)

binary_output = out.clone()
binary_output = binary_output.detach().cpu().apply_(lambda x: 0 if x < 0.5 else 1)

# sequence prediction error is calculted in bits per sequence
error = torch.sum(torch.abs(binary_output - target.detach().cpu()))

# ---logging---
print('Loss: %.2f\tError in bits per sequence: %.2f' % (loss, error))

# ---saving results---
result = {'output': binary_output, 'target': target}
print("input_size:",target.size()[0])
print("Below will show input without delimiter and reversed output and reversed target to see matching:")
input('press any key to continue:')
for i in range(binary_output.size()[0]):
    print('input: ',_input[i+1,:8],'\noutput:',binary_output[-i-1],'\ntarget:', target[-i-1],'\n')
# print("out",out)
# print("binary_output",binary_output)
# print("target",target)