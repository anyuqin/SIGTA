import numpy as np
from config import opts


class C_per_info():
    def __init__(self, p_adj: np.ndarray, p_feat: np.ndarray, p_label: np.ndarray, p_idx_train: np.ndarray,
                 p_idx_val: np.ndarray, p_idx_test: np.ndarray, p_opts: opts = opts):
        self.adj_np = p_adj
        self.feat_np = p_feat
        self.label_np = p_label
        self.idx_train = p_idx_train
        self.idx_val = p_idx_val
        self.idx_test = p_idx_test
        self.opt = p_opts
        self.random_seed = self.opt.np_random_seed

    def F_get_random_n_nodes_from_each_class(self, node_num):
        label = F_one_hot_to_label(self.label_np)
        class_num = label.max() + 1

        # initialize node_index
        node_index = np.zeros([class_num, node_num], np.int16)

        # random choose node
        for ii in range(class_num):
            temp_index_np = np.where(label == ii)[0]
            np.random.seed(self.random_seed)
            node_index[ii] = np.random.choice(temp_index_np, node_num)
        return node_index

    def F_idx_to_class(self, idx):
        """node idx -> label"""
        return np.where(self.label_np)[1][idx]

    def F_get_K_random_idx_of_single_class(self, target_class: int, node_num=10) -> np.ndarray:
        """find K nodes from one class"""
        label_not_one_hot = np.where(self.label_np)[1]
        idx_target_class = np.where(label_not_one_hot == target_class)[0]
        np.random.seed(self.random_seed)
        idx_sample_np = np.random.choice(idx_target_class, node_num)
        return idx_sample_np
        
    def F_get_K_degree_random_idx_of_single_class(self, target_class: int, node_num=10) -> np.ndarray:
        """find K nodes from one class"""
        label_not_one_hot = np.where(self.label_np)[1]
        idx_target_class = np.where(label_not_one_hot == target_class)[0]
        
        # 计算网络中所有节点的度
        degrees = self.adj_np.sum(axis=1)

        # 计算网络中所有节点的度的平均值
        avg_degree = np.mean(degrees)


        # 选择度低于平均度的节点
        idx_low_degree_nodes = [idx for idx in idx_target_class if degrees[idx]  < avg_degree]
        
        # 如果可选节点数量少于要求的节点数量，则返回所有可选节点索引
        if len(idx_low_degree_nodes) <= node_num:
            return np.array(idx_low_degree_nodes)
        else:
            # 否则，从可选节点中随机选择指定数量的节点
            np.random.seed(self.random_seed)
            idx_sample_np = np.random.choice(idx_low_degree_nodes, node_num, replace=False)
            return idx_sample_np
        
    def F_get_K_random_idx_except_one_class(self, except_class: int, node_num=10) -> np.ndarray:
        label_not_one_hot = np.where(self.label_np)[1]
        idx_target_class = np.where(label_not_one_hot != except_class)[0]
        np.random.seed(self.random_seed)
        idx_sample_np = np.random.choice(idx_target_class, node_num)
        return idx_sample_np
    def F_get_K_random_idx_from_different_communities_except_one_class(self, communities: dict, except_class: int, node_num=10) -> np.ndarray:
        label_not_one_hot = np.where(self.label_np)[1]
        
        # 收集有效的社区
        valid_nodes_by_community = {}
        
        for comm, nodes in communities.items():
            # 筛选出不属于目标类别的节点
            valid_nodes_in_comm = [node for node in nodes if label_not_one_hot[node] != except_class]
            if valid_nodes_in_comm:
                valid_nodes_by_community[comm] = valid_nodes_in_comm
        
        # 如果有效的社区数量不足，抛出错误
        if len(valid_nodes_by_community) < node_num:
            raise ValueError("有效的社区数量不足，无法满足节点数量要求。")
        

        # 按节点数对社区进行排序
        sorted_communities = sorted(valid_nodes_by_community.items(), key=lambda x: len(x[1]), reverse=True)
        # 从每个社区中选择一个或多个节点
        selected_nodes = []
        for comm, nodes in sorted_communities:
            if len(selected_nodes) >= node_num:
                break
            # 从当前社区中随机选择一个节点
            if len(nodes) > 0:
                selected_node = np.random.choice(nodes, 1)[0]
                selected_nodes.append(selected_node)
        
        # 如果选择的节点数量不足，则在剩余的节点中随机选择
        if len(selected_nodes) < node_num:
            remaining_nodes = [node for comm_nodes in valid_nodes_by_community.values() for node in comm_nodes if node not in selected_nodes]
            np.random.seed(self.random_seed)
            additional_nodes = np.random.choice(remaining_nodes, node_num - len(selected_nodes), replace=False)
            selected_nodes.extend(additional_nodes)
    
        return np.array(selected_nodes)
    def F_get_idx_except_one_class(self, except_class: int, node_num=10) -> np.ndarray:
        label_not_one_hot = np.where(self.label_np)[1]
        num_classes = np.max(label_not_one_hot) + 1  # 计算类别总数
        idx_sample_np = []
        
        # 对每个类别进行处理
        for class_idx in range(num_classes):
            if class_idx != except_class:  # 排除目标类别
                idx_target_class = np.where(label_not_one_hot == class_idx)[0]  # 获取当前类别的索引
                np.random.seed(self.random_seed)
                idx_selected = np.random.choice(idx_target_class, 3)  # 从当前类别中随机选择节点
                idx_sample_np.extend(idx_selected)  # 将选择的节点索引添加到结果列表中
        
        return np.array(idx_sample_np)

    def F_get_degree_K_random_idx_except_one_class(self, except_class: int, node_num=10) -> np.ndarray:
        label_not_one_hot = np.where(self.label_np)[1]
        unique_classes = np.unique(label_not_one_hot)

        # 初始化存储节点索引的列表
        idx_sample_list = []

        # 计算网络中所有节点的度
        degrees = self.adj_np.sum(axis=1)

        # 计算网络中所有节点的度的平均值
        average_degree = np.mean(degrees)

        # 对除了except_class之外的每个类别进行操作
        for class_label in unique_classes:
            if class_label != except_class:
                # 获取当前类别的所有节点索引
                idx_current_class = np.where(label_not_one_hot == class_label)[0]

                # 选择度高于平均值的节点
                idx_low_degree = idx_current_class[degrees[idx_current_class].flatten() > average_degree]
                
                # 将低于平均度的节点索引添加到列表中
                idx_sample_list.extend(idx_low_degree)
                
        # 随机选择节点索引
        np.random.seed(self.random_seed)
        idx_sample_np = np.random.choice(idx_sample_list, node_num)

        return idx_sample_np
    def F_get_random_idx_except_one_class(self,except_class: int, node_num=10) -> np.ndarray:
        label_not_one_hot = np.where(self.label_np)[1]  # 获取非独热编码标签
        class_labels = np.unique(label_not_one_hot)  # 所有类别的标签

        # 计算每个类别需要选取的节点数量（除了except_class）
        num_classes = len(class_labels)
        num_per_class = node_num // (num_classes - 1)
        remaining_nodes = node_num - num_per_class * (num_classes - 1)

        # 初始化存储节点索引的列表
        idx_sample_list = []

        # 对除了except_class之外的每个类别进行随机选择
        for class_label in class_labels:
            if class_label != except_class:
                idx_target_class = np.where(label_not_one_hot == class_label)[0]
                # 如果是最后一个类别，加上剩余节点
                if len(idx_sample_list) == num_classes - 2:
                    num_nodes = num_per_class + remaining_nodes
                else:
                    num_nodes = num_per_class
                # 随机选择节点索引
                np.random.seed(self.random_seed)
                idx_sample_np = np.random.choice(idx_target_class, num_nodes, replace=False)
                idx_sample_list.append(idx_sample_np)

        # 合并节点索引并返回
        idx_sample_np = np.concatenate(idx_sample_list)
        return idx_sample_np
    def F_from_label_to_idx(self, label_id: int) -> np.ndarray:
        """through label derive index"""
        label_not_one_hot = F_one_hot_to_label(self.label_np)
        idx_label_np = np.where(label_not_one_hot == label_id)[0]
        return idx_label_np


def F_one_hot_to_label(label: np.ndarray):
    """label"""
    return np.where(label)[1]
