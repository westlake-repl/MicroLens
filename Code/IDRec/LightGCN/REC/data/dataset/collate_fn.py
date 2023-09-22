import torch
import numpy as np



def seq_eval_collate(batch):
    item_seq = []
    item_target = []
    #item_length = []

    history_i = []

    for item in batch:
        history_i.append(item[0])
        item_seq.append(item[1])
        item_target.append(item[2])
        #item_length.append(item[3])
        
        
    
    
    history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_i)])
    history_i = torch.cat(history_i)
    
    item_seq = torch.tensor(item_seq)          #[batch, len]
    item_target = torch.tensor(item_target)    #[batch]
    #item_length = torch.tensor(item_length)    #[batch]    
    positive_u = torch.arange(item_seq.shape[0])   #[batch]


    #return item_seq, None, positive_u, item_target  
    return item_seq, (history_u, history_i), positive_u, item_target





def pair_eval_collate(batch):

    user = []
    history_i = []
    positive_i = []
    for item in batch:
        user.append(item[0])
        history_i.append(item[1])
        positive_i.append(item[2])
    
    user = torch.tensor(user)
    
    history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_i)])
    history_i = torch.cat(history_i)

    positive_u = torch.cat([torch.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_i)])            
    positive_i = torch.cat(positive_i)

    return user, (history_u, history_i), positive_u, positive_i





def candi_eval_collate(batch):
    item_seq = []
    item_target = []
    history_i = []

    for item in batch:
        history_i.append(item[0])
        item_seq.append(item[1])         #[n_items, len]
        item_target.append(item[2])
           
    history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_i)])
    history_i = torch.cat(history_i)
    
    item_seq = torch.stack(item_seq)          #[batch, n_items, len]
    item_target = torch.tensor(item_target)    #[batch]
    positive_u = torch.arange(item_seq.shape[0])   #[batch]

    return item_seq, (history_u, history_i), positive_u, item_target




def sampletower_train_collate(batch):
    items = []
    for item_aug in batch:
        items.append(item_aug)
    return torch.cat(items)



def base_collate(batch):
    return batch[0]

def mosampletower_train_collate(batch):
    items_index = []
    items_modal = []
    items_bias = 0
    for patch in batch:
        index= patch[0] + items_bias
        mask = patch[1]
        modal = patch[2]
        index *= mask 
        items_index.append(index)   
        items_modal.append(modal)
        items_bias += modal.shape[0]
    return torch.cat(items_index), torch.cat(items_modal)


def graph_train_collate(batch):
    inputs = []
    mask = []
    targets = []
    for patch in batch:
        inputs.append(patch[0])   
        mask.append(patch[1])
        targets.append(patch[2])
    items, n_node, A, alias_inputs = [], [], [], []
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))   #2,3,2

    max_n_node = np.max(n_node)
    for u_input in inputs:
        node = np.unique(u_input)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
    alias_inputs = torch.Tensor(np.array(alias_inputs)).long()
    items = torch.Tensor(np.array(items)).long()
    A = torch.Tensor(np.array(A)).float()
    mask = torch.Tensor(np.array(mask)).long()
    targets = torch.Tensor(np.array(targets)).long()
    return alias_inputs, A, items, mask, targets

#input_seq, masked_index, input_modal_seq, target_modal_seq  #[seq_len], [seq_len], 
#[n_node, 3, 224, 224],  [2, 3, 224, 224] 
def mograph_train_collate(batch):
    inputs = []
    mask = []
    input_modal = []
    target_modal = []    
    pad_image = torch.zeros((3,224,224))
    for patch in batch:
        inputs.append(patch[0])   
        mask.append(patch[1])
        input_modal.append(patch[2])
        target_modal.append(patch[3])
    
    items_modal, n_node, A, alias_inputs = [], [], [], []
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))   #2,3,
    max_n_node = np.max(n_node)
    for idx, u_input in enumerate(inputs):
        node = np.unique(u_input)
        items = input_modal[idx] + (max_n_node-n_node[idx])*[pad_image] + target_modal[idx] #[max_n_node+2, 3, 224, 224]
        items_modal.append(torch.stack(items)) 
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
    alias_inputs = torch.Tensor(np.array(alias_inputs)).long()
    A = torch.Tensor(np.array(A)).float()
    mask = torch.Tensor(np.array(mask)).long()
    
    items_modal = torch.cat(items_modal)  #[batch*(max_n_node+2), 3, 224, 224]
    return alias_inputs, A, mask, items_modal

#torch.tensor(history_seq), item_seq, masked_index, item_target
def graph_eval_collate(batch):
    inputs = []
    mask = []
    item_target = []
    history_i = []

    for patch in batch:
        history_i.append(patch[0])
        inputs.append(patch[1])   
        mask.append(patch[2])
        item_target.append(patch[3])

    items, n_node, A, alias_inputs = [], [], [], []
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))   #2,3,2
    max_n_node = np.max(n_node)
    for u_input in inputs:
        node = np.unique(u_input)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
    
    alias_inputs = torch.Tensor(alias_inputs).long()
    items = torch.Tensor(items).long()
    A = torch.Tensor(np.array(A)).float()
    mask = torch.Tensor(np.array(mask)).long()
        
    history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_i)])
    history_i = torch.cat(history_i)
    item_target = torch.tensor(item_target)    #[batch]   
    positive_u = torch.arange(mask.shape[0])   #[batch]
        
    return (alias_inputs, A, items, mask), (history_u, history_i), positive_u, item_target