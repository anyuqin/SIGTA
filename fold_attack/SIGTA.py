import torch as t
import numpy as np
import argparse
import os
import time
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..")) 
sys.path.append(parent_dir)
from fold_data import dataset
from config import opts
from fold_util import F_Info
from fold_util import F_Perturbation as F_per
from fold_util import F_Normalize as F_nor
from fold_util import F_Test
from fold_util import find_neighbor_idx as F_find_neighbor_idx
from fold_util import construct_sub_graph as F_construct_sub_graph
import warnings
import torch.nn.functional as F

from tqdm.auto import tqdm 
os.environ["CUDA_VISIBLE_DEVICES"]="3" 

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=bool, default=True, help="use CUDA? [True, False]")
parser.add_argument('--seed', type=int, default=30, help="random seed ")
parser.add_argument('--dataset', type=str, default="cora",
                    help="the name of dataset {\"cora\",\"pubmed\",\"citeseer\"}")
parser.add_argument('--anchor_node_num', type=int, default=3, help="the chosen anchor node number")
parser.add_argument('--fake_node_num', type=int, default=3, help="the fake node num on each anchor node")
parser.add_argument('--proxy_node_num', type=int, default=20, help="the nodes num for modifying features")
parser.add_argument('--test_time', type=int, default=1, help="multiple test number")
parser.add_argument('--target_class_id', type=int, default=0,
                    help="the target class, cora(0~6) citeseer(0~5) pubmed(0~2)")
parser.add_argument('--feature_budget', type=int, default=25, help="the budget for computing features of fake nodes")

args = parser.parse_args() 
def compute_loss(output, target_label):
    loss = F.cross_entropy(output, target_label)
    return loss

def Att_group_node_multi_anchor_nodes(opt: opts, adj, feat, label, root):
    # basic adj,feat matrix
    fake_base_adj = np.copy(adj)
    fake_base_feat = np.copy(feat) 

    fake_anchor_node_np = np.copy(opt.anchor_node_np)
    fake_proxy_node_np = np.copy(opt.proxy_node_np)

    label_anchor = np.where(label[fake_anchor_node_np[0]])[0]

    """inject non feature fake nodes"""
    
    for lp_anchor_id in range(fake_anchor_node_np.shape[0]):
        fake_base_adj, fake_base_feat = F_per.Per_add_fake_node(fake_base_adj, fake_base_feat,
                                                                fake_anchor_node_np[lp_anchor_id],
                                                                node_num=opt.fake_node_num_each_anchor_node)
    # index for fake nodes
    fake_node_idx_np = np.arange(adj.shape[0], adj.shape[0] + opt.fake_node_num)
    # index for fake anchor nodes
    fake_node_hubs = fake_node_idx_np[::opt.fake_node_num_each_anchor_node]
    
    # load model
    if opt.model == 'GCN':
        opt.model_path = "./checkpoint/{}/GCN.t7".format(opt.dataset)
    else:
        raise NotImplementedError

    model = t.load(opt.model_path)['model']

    """Records"""
    # Record : output
    record_output = np.zeros([opt.limit_fake_feat, fake_proxy_node_np.shape[0], label[0].shape[0]])
    # Record : ASR for  proxy nodes
    record_success_rate = np.zeros([opt.limit_fake_feat])
    # Record : best feature for fake nodes
    record_best_feat = np.zeros([fake_base_feat.shape[0], fake_base_adj.shape[1]])
    fake_feat_add_np = np.copy(fake_base_feat)

    # 初始化梯度记录变量
    steps_list = [2]  # 定义不同的聚合步数
    cumulative_gradients = {steps: np.zeros([opt.fake_node_num, feat.shape[1]]) for steps in steps_list}
    previous_gradients = {steps: None for steps in steps_list}

    print("STEP: Constructing Fake nodes...\n")
    # loop feature budget
    for lp_limit_feat_id in range(opt.limit_fake_feat):

        lp_proxy_id = 0
        # matrix for summing partial derivation
        # loop  proxy nodes
        for lp_proxy_idx in fake_proxy_node_np:

            # 1.1 connect  proxy node to anchor nodes
            temp_fake_adj = fake_base_adj
            for lp_anchor_idx in fake_node_hubs:
                temp_fake_adj[lp_anchor_idx][lp_proxy_idx] = 1
                temp_fake_adj[lp_proxy_idx][lp_anchor_idx] = 1

            # Extract neighbor set for the  proxy node
            temp_neighbor_set = F_find_neighbor_idx(temp_fake_adj, 2, lp_proxy_idx)

            # Create correspondence between original graph and subgraph
            proj_o_to_s = {}  # origin to sub
            proj_s_to_o = {}  # sub to origin
            for lp_set_id in range(temp_neighbor_set.shape[0]):
                proj_s_to_o[lp_set_id] = temp_neighbor_set[lp_set_id]
                proj_o_to_s[temp_neighbor_set[lp_set_id]] = lp_set_id

            # the index of  proxy node after subgraph construction
            lp_proxy_idx_proj = proj_o_to_s[lp_proxy_idx]

            # the index of fake nodes after subgraph construction
            fake_idx_proj = np.zeros(fake_node_idx_np.shape[0], dtype=np.int16)
            for lp_fake_node_id in np.arange(fake_node_idx_np.shape[0]):
                fake_idx_proj[lp_fake_node_id] = proj_o_to_s[fake_node_idx_np[lp_fake_node_id]]

            # construct subgraph based on subgraph node set
            sub_adj, sub_d, sub_feat = F_construct_sub_graph(temp_fake_adj, fake_feat_add_np, temp_neighbor_set)

            # normalize adjacency matrix adn feature matrix , Sub_adj,Sub_feat
            sub_adj_nor = F_nor.nor_sub_adj_eye(sub_adj, sub_d)
            sub_feat_nor = F_nor.normalize_feat(sub_feat)

            # to Tensor
            sub_adj_nor_T = t.from_numpy(sub_adj_nor).float()
            sub_feat_nor_T = t.from_numpy(sub_feat_nor).float()
            label_anchor_T = t.from_numpy(label_anchor).long()

            # using cuda?
            if opt.use_cuda:
                sub_adj_nor_T = sub_adj_nor_T.cuda()
                sub_feat_nor_T = sub_feat_nor_T.cuda()
                label_anchor_T = label_anchor_T.cuda()


            sub_feat_nor_T.requires_grad = True

            model.eval()
            if opt.use_cuda:
                model.cuda()


            if opt.model == "GCN":
                # GCN = Softmax(D^-0.5 * A * D^-0.5 * Relu(D^0.5 * A * D^0.5 * X) * W)
                output = model(sub_feat_nor_T, sub_adj_nor_T)
            else:
                raise NotImplementedError

            # get result
            label_proxy_T = output[[lp_proxy_idx_proj]].squeeze().argmax().unsqueeze(dim=0)

            # 计算交叉熵损失
            target_label = label_anchor_T.expand_as(label_proxy_T)
            loss = compute_loss(output[lp_proxy_idx_proj].unsqueeze(0), target_label)
            loss.backward()

            # acquire grad
            temp_feat_grad  = sub_feat_nor_T.grad.cpu().detach().numpy()
            output_proxy = output[lp_proxy_idx_proj].cpu().detach().numpy()
            
            current_gradient = temp_feat_grad[fake_idx_proj]
            # cumulative_gradients[steps] += current_gradient
            for steps in steps_list:
                cumulative_gradients[steps] += current_gradient

            # sum grad
         
            record_output[lp_limit_feat_id, lp_proxy_id] = output_proxy

            lp_proxy_id = lp_proxy_id + 1

            # undo the link between anchor node and  proxy node
            for lp_anchor_idx in fake_node_hubs:
                temp_fake_adj[lp_anchor_idx][lp_proxy_idx] = 0   
                temp_fake_adj[lp_proxy_idx][lp_anchor_idx] = 0
        
        for steps in steps_list:
            if (lp_limit_feat_id + 1) % steps == 0:
                previous_gradients[steps] = np.copy(cumulative_gradients[steps])
                fake_feat_add_np = F_per.Per_add_fake_feat_based_on_grad_multi_anchor_nodes(cumulative_gradients[steps], fake_feat_add_np)
                cumulative_gradients[steps] = np.zeros([opt.fake_node_num, feat.shape[1]]) 

        # calculate the ASR on  proxy node set
        anchor_success_rate = F_Test.Test_attack_success_rate_for_Class_Node(label, record_output[lp_limit_feat_id],
                                                                             fake_anchor_node_np[0])
        # record the ASR 
        record_success_rate[lp_limit_feat_id] = anchor_success_rate

        # record the best iteration
        if lp_limit_feat_id == 0:
            record_best_feat = np.copy(fake_feat_add_np)
            record_best_iter = lp_limit_feat_id
        elif record_success_rate.max() <= anchor_success_rate:
            record_best_feat = np.copy(fake_feat_add_np)
            record_best_iter = lp_limit_feat_id

    """Test Step"""

    # find the victim node set
    victim_node_idx = np.arange(adj.shape[0])

    # find the target class
    label_anchor = label_not_one_hot[fake_anchor_node_np[0]]
    # exclude the nodes that belong to the target class
    victim_node_idx = np.setdiff1d(victim_node_idx, np.where(label_not_one_hot == label_anchor))
    # exclude the  proxy nodes
    victim_node_idx = np.setdiff1d(victim_node_idx, fake_proxy_node_np)

    # test node limitation
    if victim_node_idx.shape[0] > 2000:
        victim_node_idx = np.random.choice(victim_node_idx, 2000)

    temp_record = 0
    print("STEP: Testing on victim node set \n")
    # go through all the test nodes
    for lp_victim_node_id in tqdm(range(victim_node_idx.shape[0]), position=0, leave=True, ncols=80):
        lp_victim_node_idx = victim_node_idx[lp_victim_node_id]

        # link the victim node and anchor nodes
        temp_test_adj = fake_base_adj
       
        for lp_anchor_idx in fake_node_hubs:
            temp_test_adj[lp_anchor_idx, lp_victim_node_idx] = 1
            temp_fake_adj[lp_victim_node_idx][lp_anchor_idx] = 1
        # extract the neighbor of victim node
        test_neighbor_set = F_find_neighbor_idx(temp_test_adj, 2, lp_victim_node_idx)

        # Create correspondence between original graph and subgraph
        test_proj_o_to_s = {}
        test_proj_s_to_o = {}
        for lp_set_id in range(test_neighbor_set.shape[0]):
            test_proj_s_to_o[lp_set_id] = test_neighbor_set[lp_set_id]
            test_proj_o_to_s[test_neighbor_set[lp_set_id]] = lp_set_id

        lp_test_idx_proj = test_proj_o_to_s[lp_victim_node_idx]

        # construct subgraph for victim node based on neighbor set
        sub_adj, sub_d, sub_feat = F_construct_sub_graph(temp_test_adj, fake_feat_add_np, test_neighbor_set)

        # normalize subgraph adjacency matrix and feature matrix
        test_sub_adj_nor = F_nor.nor_sub_adj_eye(sub_adj, sub_d)
        test_sub_feat_nor = F_nor.normalize_feat(sub_feat)

        # get the output
        sub_adj_nor_T = t.from_numpy(test_sub_adj_nor).float()
        sub_feat_nor_T = t.from_numpy(test_sub_feat_nor).float()

        # use cuda?
        if opt.use_cuda:
            sub_adj_nor_T = sub_adj_nor_T.cuda()
            sub_feat_nor_T = sub_feat_nor_T.cuda()


        model.eval()
        if opt.use_cuda:
            model.cuda()

        if opt.model == "GCN":
            output = model(sub_feat_nor_T, sub_adj_nor_T)
        else:
            raise NotImplementedError

        # the label of victim label after anchor
        label_proxy_T = output[lp_test_idx_proj].argmax().item()

        if label_proxy_T == label_anchor:
            temp_record = temp_record + 1

        # undo the connection between victim node and anchor node
        
        for lp_anchor_idx in fake_node_hubs:
            temp_test_adj[lp_anchor_idx, lp_victim_node_idx] = 0
            temp_test_adj[lp_victim_node_idx, lp_anchor_idx] = 0
    # AVG asr on victim node set
    temp_success_rate = temp_record / victim_node_idx.shape[0]

    temp_log = "\nThe attack success rate on victim node set：{}\n".format(temp_success_rate)
    print(temp_log)
    with open("./logs/{}_{}_iter2_fakenode{}_class{}.txt".format(opt.dataset,opt.anchor_node_num,args.fake_node_num,args.target_class_id), 'a+') as f:
        f.write(temp_log)

if __name__ == '__main__':

    model_path = "../checkpoint"

    opt = opts()
    opt.data_path = r"./fold_data/Data/Planetoid"
    opt.model_path = r"./checkpoint"
    opt.dataset = args.dataset  # dataset name
    opt.model = 'GCN'
    opt.use_cuda = args.cuda
    opt.limit_fake_feat = args.feature_budget  # feature budget
    opt.fake_node_num_each_anchor_node = args.fake_node_num  # fake node number
    opt.anchor_node_num = args.anchor_node_num  # anchor node number
    opt. proxy_node_num = args.proxy_node_num  #  proxy node number
    opt.total_test_time = args.test_time  # multiple tests
    opt.np_random_seed = args.seed
    opt.target_class = args.target_class_id
    opt.fake_node_num = opt.fake_node_num_each_anchor_node * opt.anchor_node_num

    # Loading dataset
    data_load = dataset.c_dataset_loader(opt.dataset, opt.data_path)
    base_adj, base_feat, label, idx_train, idx_val, idx_test = data_load.process_data()
    label_not_one_hot = F_Info.F_one_hot_to_label(label)
   
    data_info = F_Info.C_per_info(base_adj, base_feat, label, idx_train, idx_val, idx_test, opt)

    # check_input
    if opt.target_class < 0 or opt.target_class > label_not_one_hot.max():
        print("invalid class id")
        raise ValueError

    # multi proxy
    proxy = [opt. proxy_node_num]

    # class_num_set
    class_num = {}
    class_num['cora'] = 7
    class_num['citeseer'] = 6
    class_num['pubmed'] = 3

    # randomly choose anchor node and  proxy node
    anchor_node_np = np.zeros([opt.total_test_time, opt.anchor_node_num]).astype(np.int16)
    proxy_node_np = np.zeros([opt.total_test_time, opt. proxy_node_num]).astype(np.int16)
  
    # pick up different set of nodes for multiple tests
    for lp_random_seed in np.arange(opt.np_random_seed, opt.np_random_seed + opt.total_test_time):
        data_info.random_seed = lp_random_seed
        #   pick anchor nodes
        anchor_node_np[lp_random_seed - opt.np_random_seed] = data_info.F_get_K_random_idx_of_single_class(
            opt.target_class,
            opt.anchor_node_num)
        #   pick  proxy nodes
        proxy_node_np[lp_random_seed - opt.np_random_seed] = data_info.F_get_K_random_idx_except_one_class(
            opt.target_class, opt. proxy_node_num)
    
    unique_classes, counts = np.unique(label_not_one_hot[ proxy_node_np], return_counts=True)
    
    # 创建字典存储每个类别的数目
    class_counts = dict(zip(unique_classes, counts))
    print(class_counts)
    
    warnings.filterwarnings("ignore")
    opt.temp_test_time = 0
    for lp_test_time in range(opt.total_test_time):
        opt.anchor_node_np = anchor_node_np[lp_test_time]
        opt. proxy_node_np =  proxy_node_np[lp_test_time]

        temp_log = "Runing on {}，Class {}， Test {}.... {} anchor nodes...{} fake nodes each...{} proxy nodes...\n".format(
            opt.dataset, opt.target_class,
            lp_test_time,
            opt.anchor_node_num,
            opt.fake_node_num_each_anchor_node, opt.proxy_node_num)
        print(temp_log)

        if not os.path.isdir("./logs"):
            os.makedirs("./logs")
        with open("./logs/{}_{}_iter2_fakenode{}_class{}.txt".format(opt.dataset,opt.anchor_node_num,args.fake_node_num,args.target_class_id), 'a+') as f:
            f.write("{}\n".format(temp_log))
        #  time.strftime("%Y%m%d"),
        Att_group_node_multi_anchor_nodes(opt, adj=base_adj, feat=base_feat, label=label,
                                          root="./fold_result/{}/{}".format(opt.model, opt.dataset))
        opt.temp_test_time = opt.temp_test_time + 1
