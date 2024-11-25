import numpy as np
import networkx as nx


def Per_add_fake_node(adj_np: np.ndarray, feat_np: np.ndarray, attack_node_index: int, node_num=1):
    """
    Add fake nodes to the attack nodes while maintaining similar degree distribution.
    :param adj_np: Original adjacency matrix
    :param feat_np: Original feature matrix
    :param fake_attack_node_np: Index of the node to which fake nodes should be connected
    :param node_num: Number of fake nodes to add
    :return: adj_fake_np, feat_fake_np
    主节点连接
    """

    shape = adj_np.shape[0]
    # Initialize adjacency matrix
    adj_fake_np = np.zeros((shape + node_num, shape + node_num))
    adj_fake_np[:shape, :shape] = adj_np

    # Initialize feature matrix 
    feat_num = feat_np.shape[1]
    feat_fake_np = np.zeros((shape + node_num, feat_num))
    feat_fake_np[:shape, :feat_num] = feat_np
    
    # add link between  fake nodes
    for i in range(shape + 1, shape + node_num):
            adj_fake_np[shape, i] = 1
            adj_fake_np[i, shape] = 1

    # Connect fake nodes to the specified node fake_attack_node_np
    adj_fake_np[attack_node_index, shape] = 1
    adj_fake_np[shape, attack_node_index] = 1

    return adj_fake_np, feat_fake_np

def Per_add_Random_fake_node(adj_np: np.ndarray, feat_np: np.ndarray, attack_node_index: int, node_num=1):
    """
    Add fake nodes to the attack nodes while maintaining similar degree distribution.
    :param adj_np: Original adjacency matrix
    :param feat_np: Original feature matrix
    :param fake_attack_node_np: Index of the node to which fake nodes should be connected
    :param node_num: Number of fake nodes to add
    :return: adj_fake_np, feat_fake_np
    """

    
    shape = adj_np.shape[0]
    # Initialize adjacency matrix
    adj_fake_np = np.zeros((shape + node_num, shape + node_num))
    adj_fake_np[:shape, :shape] = adj_np

    # Initialize feature matrix 
    feat_num = feat_np.shape[1]
    feat_fake_np = np.zeros((shape + node_num, feat_num))
    feat_fake_np[:shape, :feat_num] = feat_np
    
    # Randomly connect fake nodes ensuring no isolated nodes
    edges = []
    for i in range(node_num):
        for j in range(i + 1, node_num):
            if np.random.rand() < 0.5:
                edges.append((i, j))

    # Ensure no isolated nodes
    for i in range(node_num):
        if not any(i in edge for edge in edges):
            j = np.random.choice([x for x in range(node_num) if x != i])
            edges.append((i, j))

    # Add edges to the adjacency matrix
    for i, j in edges:
        adj_fake_np[shape + i, shape + j] = 1
        adj_fake_np[shape + j, shape + i] = 1
 
    # Connect fake nodes to the specified node fake_attack_node_np
    adj_fake_np[attack_node_index, shape] = 1
    adj_fake_np[shape, attack_node_index] = 1

    return adj_fake_np, feat_fake_np


def Per_add_fake_feat_based_on_grad_multi_anchor_nodes(p_grad_np: np.ndarray, p_feat_fake: np.ndarray):
    """
    according to the derivation, modify feature of fake nodes
    """

    # initialize
    fake_node_num = p_grad_np.shape[0]
    feat_num = p_grad_np.shape[1]
    total_node_num = p_feat_fake.shape[0]
    feat_fake = np.copy(p_feat_fake)

    """find the Grad(i,j)"""
    
    # grad_exclude_sum_sort = np.unique(np.sort(p_grad_np.flatten()))
    grad_exclude_sum_sort = np.unique(np.sort(p_grad_np.flatten()))

    # find the max, if not, check next maxi value
    for ii in range(50):
        
        temp_feat_arg = grad_exclude_sum_sort[ii]
        temp_feat_idx_np = np.where(p_grad_np.flatten() == temp_feat_arg)[0]

        # check if there are same grad value
        for jj in range(temp_feat_idx_np.shape[0]):
            temp_feat_node_idx = int(temp_feat_idx_np[jj] / feat_num) + total_node_num - fake_node_num
            temp_feat_idx = int(temp_feat_idx_np[jj] % feat_num)   
            #
            if feat_fake[temp_feat_node_idx, temp_feat_idx] == 1:
                continue
            else:
                feat_fake[temp_feat_node_idx, temp_feat_idx] = 1
                return feat_fake

    return feat_fake









