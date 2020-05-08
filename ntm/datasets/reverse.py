import torch
import json
import torch
from torch.utils.data import Dataset
from torch.distributions.binomial import Binomial


class ReverseDataset(Dataset):
    """A Dataset class to generate random examples for the reverse task. Each
    sequence has a random length between `min_seq_len` and `max_seq_len`.
    Each vector in the sequence has a fixed length of `seq_width`. The vectors
    are bounded by start and end delimiter flags.

    To account for the delimiter flags, the input sequence length as well
    width is two more than the target sequence.
    """

    def __init__(self, task_params): 
        """Initialize a dataset instance for reverse task.

        Arguments
        ---------
        task_params : dict
            A dict containing parameters relevant to reverse task.
        """
        self.seq_width = task_params['seq_width']
        self.min_seq_len = task_params['min_seq_len']
        self.max_seq_len = task_params['max_seq_len']
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.to(self.device)

    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536

    def __getitem__(self, idx):
        # idx only acts as a counter while generating batches.
        seq_len = torch.randint(
            self.min_seq_len, self.max_seq_len, (1,), dtype=torch.long).item()
        prob = 0.5 * torch.ones([seq_len, self.seq_width], dtype=torch.float64)
        seq = Binomial(1, prob).sample()
        # seq_len = torch.randint(
        #     self.min_seq_len, self.max_seq_len, (1,), dtype=torch.long).item().to(self.device)
        # prob = 0.5 * torch.ones([seq_len, self.seq_width], dtype=torch.float64).to(self.device)
        # seq = Binomial(1, prob).sample().to(self.device)

        # inverse the input
        idx = [i for i in range(seq.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        inverted_tensor = seq.index_select(0, idx)

        # fill in input sequence, two bit longer and wider than target
        # input_seq = torch.zeros([seq_len + 2, self.seq_width + 2]).to(self.device)
        input_seq = torch.zeros([seq_len + 2, self.seq_width + 2])
        input_seq[0, self.seq_width] = 1.0  # start delimiter
        input_seq[1:seq_len + 1, :self.seq_width] = seq
        input_seq[seq_len + 1, self.seq_width + 1] = 1.0  # end delimiter

        # target_seq = torch.zeros([seq_len, self.seq_width]).to(self.device)
        target_seq = torch.zeros([seq_len, self.seq_width])
        target_seq[:seq_len, :self.seq_width] = inverted_tensor
        return {'input': input_seq, 'target': target_seq}

if __name__ == "__main__":
    task_json = './ntm/tasks/reverse.json'
    task_params = json.load(open(task_json))
    dataset=ReverseDataset(task_params)
    dataset[0]