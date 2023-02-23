import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet

from transformers import BertModel, BertConfig


# START ADDING SOFT DECISION TREE ########################

import os
import time

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class InnerNode():

    def __init__(self, depth, args):
        self.args = args
        self.fc = nn.Linear(self.args.input_dim, 1)
        beta = torch.randn(1)
        #beta = beta.expand((self.args.batch_size, 1))
        if self.args.cuda:
            beta = beta.cuda()
        self.beta = nn.Parameter(beta)
        self.leaf = False
        self.prob = None
        self.leaf_accumulator = []
        self.lmbda = self.args.lmbda * 2 ** (-depth)
        self.build_child(depth)
        self.penalties = []

    def reset(self):
        self.leaf_accumulator = []
        self.penalties = []
        self.left.reset()
        self.right.reset()

    def build_child(self, depth):
        if depth < self.args.max_depth:
            self.left = InnerNode(depth+1, self.args)
            self.right = InnerNode(depth+1, self.args)
        else :
            self.left = LeafNode(self.args)
            self.right = LeafNode(self.args)

    def forward(self, x):
        return(F.sigmoid(self.beta*self.fc(x)))
    
    def select_next(self, x):
        prob = self.forward(x)
        if prob < 0.5:
            return(self.left, prob)
        else:
            return(self.right, prob)

    def cal_prob(self, x, path_prob):
        self.prob = self.forward(x) #probability of selecting right node
        self.path_prob = path_prob
        left_leaf_accumulator = self.left.cal_prob(x, path_prob * (1-self.prob))
        right_leaf_accumulator = self.right.cal_prob(x, path_prob * self.prob)
        self.leaf_accumulator.extend(left_leaf_accumulator)
        self.leaf_accumulator.extend(right_leaf_accumulator)
        return(self.leaf_accumulator)

    def get_penalty(self):
        penalty = (torch.sum(self.prob * self.path_prob) / torch.sum(self.path_prob), self.lmbda)
        if not self.left.leaf:
            left_penalty = self.left.get_penalty()
            right_penalty = self.right.get_penalty()
            self.penalties.append(penalty)
            self.penalties.extend(left_penalty)
            self.penalties.extend(right_penalty)
        return(self.penalties)


class LeafNode():
    def __init__(self, args):
        self.args = args
        self.param = torch.randn(self.args.output_dim)
        if self.args.cuda:
            self.param = self.param.cuda()
        self.param = nn.Parameter(self.param)
        self.leaf = True
        self.softmax = nn.Softmax()

    def forward(self):
        return(self.softmax(self.param.view(1,-1)))

    def reset(self):
        pass

    def cal_prob(self, x, path_prob):
        Q = self.forward()
        #Q = Q.expand((self.args.batch_size, self.args.output_dim))
        Q = Q.expand((path_prob.size()[0], self.args.output_dim))
        return([[path_prob, Q]])


class SoftDecisionTree(nn.Module):

    def __init__(self, args):
        super(SoftDecisionTree, self).__init__()
        self.args = args
        self.root = InnerNode(1, self.args)
        self.collect_parameters() ##collect parameters and modules under root node
        self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.test_acc = []
        self.define_extras(self.args.batch_size)
        self.best_accuracy = 0.0

    def define_extras(self, batch_size):
        ##define target_onehot and path_prob_init batch size, because these need to be defined according to batch size, which can be differ
        self.target_onehot = torch.FloatTensor(batch_size, self.args.output_dim)
        self.target_onehot = Variable(self.target_onehot)
        self.path_prob_init = Variable(torch.ones(batch_size, 1))
        if self.args.cuda:
            self.target_onehot = self.target_onehot.cuda()
            self.path_prob_init = self.path_prob_init.cuda()
    def forward(self, x):
        node = self.root
        path_prob = Variable(torch.ones(self.args.batch_size, 1))
        while not node.leaf:
            node, prob = node.select_next(x)
            path_prob *= prob
        return node()      
    def cal_loss(self, x, y):
        batch_size = y.size()[0]
        leaf_accumulator = self.root.cal_prob(x, self.path_prob_init)
        loss = 0.
        max_prob = [-1. for _ in range(batch_size)]
        max_Q = [torch.zeros(self.args.output_dim) for _ in range(batch_size)]
        for (path_prob, Q) in leaf_accumulator:
            TQ = torch.bmm(y.view(batch_size, 1, self.args.output_dim), torch.log(Q).view(batch_size, self.args.output_dim, 1)).view(-1,1)
            loss += path_prob * TQ
            path_prob_numpy = path_prob.cpu().data.numpy().reshape(-1)
            for i in range(batch_size):
                if max_prob[i] < path_prob_numpy[i]:
                    max_prob[i] = path_prob_numpy[i]
                    max_Q[i] = Q[i]
        loss = loss.mean()
        penalties = self.root.get_penalty()
        C = 0.
        for (penalty, lmbda) in penalties:
            C -= lmbda * 0.5 *(torch.log(penalty) + torch.log(1-penalty))
        output = torch.stack(max_Q)
        self.root.reset() ##reset all stacked calculation
        return(-loss + C, output) ## -log(loss) will always output non, because loss is always below zero. I suspect this is the mistake of the paper?

    def collect_parameters(self):
        nodes = [self.root]
        self.module_list = nn.ModuleList()
        self.param_list = nn.ParameterList()
        while nodes:
            node = nodes.pop(0)
            if node.leaf:
                param = node.param
                self.param_list.append(param)
            else:
                fc = node.fc
                beta = node.beta
                nodes.append(node.right)
                nodes.append(node.left)
                self.param_list.append(beta)
                self.module_list.append(fc)

    def train_(self, train_loader, epoch):
        self.train()
        self.define_extras(self.args.batch_size)
        for batch_idx, (data, target) in enumerate(train_loader):
            correct = 0
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            #data = data.view(self.args.batch_size,-1)
            target = Variable(target)
            target_ = target.view(-1,1)
            batch_size = target_.size()[0]
            data = data.view(batch_size,-1)
            ##convert int target to one-hot vector
            data = Variable(data)
            if not batch_size == self.args.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)
            self.target_onehot.data.zero_()            
            self.target_onehot.scatter_(1, target_, 1.)
            self.optimizer.zero_grad()

            loss, output = self.cal_loss(data, self.target_onehot)
            #loss.backward(retain_variables=True)
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            accuracy = 100. * correct / len(data)

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0],
                    correct, len(data),
                    accuracy))

    def test_(self, test_loader, epoch):
        self.eval()
        self.define_extras(self.args.batch_size)
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            target = Variable(target)
            target_ = target.view(-1,1)
            batch_size = target_.size()[0]
            data = data.view(batch_size,-1)
            ##convert int target to one-hot vector
            data = Variable(data)
            if not batch_size == self.args.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)
            self.target_onehot.data.zero_()            
            self.target_onehot.scatter_(1, target_, 1.)
            _, output = self.cal_loss(data, self.target_onehot)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
        accuracy = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(
            correct, len(test_loader.dataset),
            accuracy))
        self.test_acc.append(accuracy)

        if accuracy > self.best_accuracy:
            self.save_best('./result')
            self.best_accuracy = accuracy

    def save_best(self, path):
        try:
            os.makedirs('./result')
        except:
            print('directory ./result already exists')

        with open(os.path.join(path, 'best_model.pkl'), 'wb') as output_file:
            pickle.dump(self, output_file)
            
# END ADDING SOFT DECISION TREE ########################

class MMIM(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args: 
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.hp = hp
        self.add_va = hp.add_va
        hp.d_tout = hp.d_tin
        self.soft_tree = SoftDecisionTree()

        self.text_enc = LanguageEmbeddingLayer(hp)
        self.visual_enc = RNNEncoder(
            in_size = hp.d_vin,
            hidden_size = hp.d_vh,
            out_size = hp.d_vout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )
        self.acoustic_enc = RNNEncoder(
            in_size = hp.d_ain,
            hidden_size = hp.d_ah,
            out_size = hp.d_aout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )

        # For MI maximization
        self.mi_tv = MMILB(
            x_size = hp.d_tout,
            y_size = hp.d_vout,
            mid_activation = hp.mmilb_mid_activation,
            last_activation = hp.mmilb_last_activation
        )

        self.mi_ta = MMILB(
            x_size = hp.d_tout,
            y_size = hp.d_aout,
            mid_activation = hp.mmilb_mid_activation,
            last_activation = hp.mmilb_last_activation
        )

        if hp.add_va:
            self.mi_va = MMILB(
                x_size = hp.d_vout,
                y_size = hp.d_aout,
                mid_activation = hp.mmilb_mid_activation,
                last_activation = hp.mmilb_last_activation
            )

        dim_sum = hp.d_aout + hp.d_vout + hp.d_tout

        # CPC MI bound
        self.cpc_zt = CPC(
            x_size = hp.d_tout, # to be predicted
            y_size = hp.d_prjh,
            n_layers = hp.cpc_layers,
            activation = hp.cpc_activation
        )
        self.cpc_zv = CPC(
            x_size = hp.d_vout,
            y_size = hp.d_prjh,
            n_layers = hp.cpc_layers,
            activation = hp.cpc_activation
        )
        self.cpc_za = CPC(
            x_size = hp.d_aout,
            y_size = hp.d_prjh,
            n_layers = hp.cpc_layers,
            activation = hp.cpc_activation
        )

        # Trimodal Settings
        self.fusion_prj = SubNet(
            in_size = dim_sum,
            hidden_size = hp.d_prjh,
            n_class = hp.n_class,
            dropout = hp.dropout_prj
        )
            
    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None, mem=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
        text = enc_word[:,0,:] # (batch_size, emb_size)

        acoustic = self.acoustic_enc(acoustic, a_len)
        visual = self.visual_enc(visual, v_len)

        if y is not None:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual, labels=y, mem=mem['tv'])
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic, labels=y, mem=mem['ta'])
            # for ablation use
            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic, labels=y, mem=mem['va'])
        else:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual)
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic)
            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic)


        # Linear proj and pred
        fusion, preds = self.fusion_prj(torch.cat([text, acoustic, visual], dim=1))
        preds = self.soft_tree(fusion)

        nce_t = self.cpc_zt(text, fusion)
        nce_v = self.cpc_zv(visual, fusion)
        nce_a = self.cpc_za(acoustic, fusion)
        
        nce = nce_t + nce_v + nce_a

        pn_dic = {'tv':tv_pn, 'ta':ta_pn, 'va': va_pn if self.add_va else None}
        lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)
        H = H_tv + H_ta + (H_va if self.add_va else 0.0)

        return lld, nce, preds, pn_dic, H
