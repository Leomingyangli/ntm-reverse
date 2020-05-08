Neural Turing Machines (Pytorch)
=================================
Code for the paper

**[Neural Turing Machines][1]**

Alex Graves, Greg Wayne, Ivo Danihelka

[1]: https://arxiv.org/abs/1410.5401
Neural Turing Machines (NTMs) contain a recurrent network coupled with an external memory resource, which it can interact with by attentional processes. Therefore NTMs can be called Memory Augmented Neural Networks. They are end-to-end differentiable and thus are hypothesised at being able to learn simple algorithms. They outperform LSTMs in learning several algorithmic tasks due to the presence of external memory without an increase in parameters and computation.

This repository is a stable Pytorch implementation of a Neural Turing Machine and contains the code for training, evaluating and visualizing results for the Reverse task.

<p align="center">
<img width="500" height="300" src="https://www.researchgate.net/profile/Gabriel_Makdah/publication/279864730/figure/fig3/AS:372237233344513@1465759680918/Neural-Turing-Machine-architecture-The-controller-or-neural-network-receives-the-input.png">
</p>
<p align="center">
<em> Neural Turing Machine architecture </em>
</p>

Setup
=================================
This code is implemented in Pytorch 1.5.0 and Python >=3.5. To setup, proceed as follows :

To install Pytorch head over to ```https://pytorch.org/``` or install using miniconda or anaconda package by running 
```conda install -c soumith pytorch ```.

Clone this repository :

```
git clone https://www.github.com/kdexd/ntm-pytorch
```

The other python libraries that you'll need to run the code :
```
pip install numpy 
pip install tensorboardX
pip install matplotlib
pip install tqdm
pip install Pillow
```

Training
================================
Training works with default arguments by :
```
python train.py
```
There are totally 9 combinations between 3 network configurations(m1,m2,m3) and 3 training data configurations(d1,d2,d3). An example about training some specific combination by:
```
python train.py -config m1d1
```
The script runs with all arguments set to default value. If you wish to changes any of these, run the script with ```-h``` to see available arguments and change them as per need be.
```
usage : train.py [-h] [-task_json TASK_JSON] [-batch_size BATCH_SIZE] 
                [-num_iters NUM_ITERS] [-config CONFIG] [-lr LR] [-momentum MOMENTUM]
                [-alpha ALPHA] [-saved_model SAVED_MODEL] [-beta1 BETA1] [-beta2 BETA2]
```
Both RMSprop and Adam optimizers have been provided. ```-momentum``` and ```-alpha``` are parameters for RMSprop and ```-beta1``` and ```-beta2``` are parameters for Adam. All these arguments are initialized to their default values.


Evaluation
===============================
The model was trained and was evaluated as mentioned in the paper. The results were in accordance with the paper. Saved models for all the tasks are available in the ```saved_model``` folder. The model for reverse task has been trained upto 100k iterations. The code for saving and loading the model has been incorporated in ```train.py``` and ```evaluate.py``` respectively.

The evaluation parameters for all tasks have been included in ```evaluate.py```.

Evaluation can be done as follows :
```
python evaluate.py -config m1d1
```

Visualization
===============================
I have integrated Tensorboardx to visualize training and evaluation loss. All visualized data have been provided in the ```runs``` folder. To install tensorboard logger use :
```
pip install tensorboardx
```
Sample outputs and loss have been provided in the ```log``` folder.

Acknowledgements
===============================
- I have used the following codebase as a reference for my implementation : **[vlgiitr/ntm-pytorch][2]**  

[2]:https://github.com/vlgiitr/ntm-pytorch

