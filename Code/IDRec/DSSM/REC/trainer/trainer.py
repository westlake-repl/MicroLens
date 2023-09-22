import os
from logging import getLogger
from time import time
import time as t
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm


from REC.data.dataset import BatchDataset
from torch.utils.data import DataLoader
from REC.evaluator import Evaluator, Collector
from REC.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
get_tensorboard, set_color, get_gpu_usage, WandbLogger


class Trainer(object):
    def __init__(self, config, model):
        super(Trainer, self).__init__()
        self.config = config
        self.model = model
        self.logger = getLogger()
        
        self.wandblogger = WandbLogger(config)
     
        self.optim_args = config['optim_args']
        # 为了调参：
        # for k, v in self.optim_args.items():
        #     self.optim_args[k] = config[f'args_{k}']

        if not self.optim_args:
            self.learning_rate = config['learning_rate']
            self.weight_decay = config['weight_decay']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']

        self.rank = torch.distributed.get_rank()
        
        if self.rank == 0:          
            self.tensorboard = get_tensorboard(self.logger)
    
        self.checkpoint_dir = config['checkpoint_dir']
        if self.rank == 0:
            ensure_dir(self.checkpoint_dir)
                
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        
        self.use_modality = config['use_modality']
            
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
     
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_feature = None
        self.tot_item_num = None

               
    def _build_optimizer(self):
        if self.optim_args:
            params = self.model.named_parameters()
            modal_params = []
            recsys_params = []
            modal_decay_params = []
            recsys_decay_params = []
            decay_check_name = self.config['check_decay_name']  #一般设为None
            for index, (name, param) in enumerate(params):
                if param.requires_grad:
                    if 'visual_encoder' in name:
                        modal_params.append(param)
                    else:
                        recsys_params.append(param)                                        
                    if decay_check_name:
                        if decay_check_name in name:
                            modal_decay_params.append(param)
                        else :
                            recsys_decay_params.append(param)            
            if decay_check_name:
                optimizer = optim.AdamW([
                    {'params': modal_decay_params, 'lr': self.optim_args['modal_lr'],'weight_decay':self.optim_args['modal_decay']},
                    {'params': recsys_decay_params, 'lr': self.optim_args['rec_lr'],'weight_decay':self.optim_args['rec_decay']}
                ])
                optim_output = set_color(f'recsys_decay_params_len: {len(recsys_decay_params)}  modal_params_decay_len: {len(modal_decay_params)}', 'blue')            
                self.logger.info(optim_output)
            else:
                optimizer = optim.AdamW([
                    {'params': modal_params, 'lr': self.optim_args['modal_lr'],'weight_decay':self.optim_args['modal_decay']},
                    {'params': recsys_params, 'lr': self.optim_args['rec_lr'],'weight_decay':self.optim_args['rec_decay']}
                ])
                optim_output = set_color(f'recsys_lr_params_len: {len(recsys_params)}  modal_lr_params_len: {len(modal_params)}', 'blue')            
                self.logger.info(optim_output)

        else :  
            params = self.model.parameters() 
            optimizer = optim.AdamW(params, lr=self.learning_rate,weight_decay=self.weight_decay)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False): 
        self.model.train()
        #self.model.module.visual_encoder.eval()
        total_loss = 0

        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=85,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
  
        for batch_idx, data in enumerate(iter_data):
            # self.logger.info(data)
            # import sys
            # if batch_idx > 5:
            #     sys.exit()
            # if batch_idx < 2764:
            #     continue            
            self.optimizer.zero_grad()
            data = self.to_device(data)           
            losses = self.model(data)            
            self._check_nan(losses)
            total_loss = total_loss + losses.item()
            losses.backward()

            #
            # parameters = self.model.parameters()
            # parameters = [p for p in parameters if p.grad is not None]
            # device = parameters[0].grad.device
            # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2)
            # if batch_idx % 100 == 0 or total_norm.item() > 1:
            #     self.logger.info(batch_idx)
            #     self.logger.info(total_norm)
            #
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

            # if batch_idx == 0:
            #     param1 = self.model.module.user_embedding.weight.detach().clone()
            #     param2 = self.model.module.item_embedding.weight.detach().clone()
            # elif batch_idx == 1:
            #     param3 = self.model.module.user_embedding.weight.detach().clone()
            #     param4 = self.model.module.item_embedding.weight.detach().clone()
            
            #     print((param1!= param3).sum())
            #     print((param2!= param4).sum())
            #     print((param1!=torch.inf).sum())
            #     print((param2!=torch.inf).sum())
            #     import sys
            #     sys.exit()
        return total_loss

    
    def _valid_epoch(self, valid_data, show_progress=False):
        torch.distributed.barrier()
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        torch.distributed.barrier()
        return valid_score, valid_result

    def _save_checkpoint(self, epoch, verbose=True):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        if self.rank == 0 :
            saved_model_file = self.saved_model_file
            state = {
                'config': self.config,
                'epoch': epoch,
                'cur_step': self.cur_step,
                'best_valid_score': self.best_valid_score,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),   # torch的随机数状态
                'cuda_rng_state': torch.cuda.get_rng_state()
            }
            torch.save(state, saved_model_file)
            if verbose:
                self.logger.info(set_color('Saving current', 'blue') + f': {saved_model_file}')
        torch.distributed.barrier()
        
        

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        #self.saved_model_file = resume_file
        checkpoint = torch.load(resume_file,map_location=torch.device('cpu'))
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            self.logger.warning(
                'Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
     
        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng_state'])  # load torch的随机数生成器状态
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])  # load torch.cuda的随机数生成器状态
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            'learning_rate': self.config['learning_rate'],
            'weight_decay': self.config['weight_decay'],
            'train_batch_size': self.config['train_batch_size']
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values() for parameter in parameters
        }.union({'model', 'dataset', 'config_files', 'device'})
        # other model-specific hparam
        hparam_dict.update({
            para: val
            for para, val in self.config.final_config_dict.items() if para not in unrecorded_parameter
        })
        for k in hparam_dict:
            k = k.replace('@','_')
            if hparam_dict[k] is not None and not isinstance(hparam_dict[k], (bool, str, float, int)):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(hparam_dict, {'hparam/best_valid_result': best_valid_result})

    
    def to_device(self, data):
        device  = self.device
        if isinstance(data, tuple) or isinstance(data, list):        
            tdata = ()
            for d in data:
                d = d.to(device)
                tdata += (d,)
            return tdata
        elif isinstance(data, dict):
            for k, v in data.items():
                data[k] = v.to(device)
            return data
        else:
            return data.to(device) 

    
    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)
        #self.eval_collector.data_collect(train_data.dataset.dataload)    
        valid_step = 0
        for epoch_idx in range(self.start_epoch, self.epochs):
            #train  
            if self.config['need_training'] == None or self.config['need_training']:              
                train_data.sampler.set_epoch(epoch_idx)
                training_start_time = time()
                train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
                self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
                if verbose:
                    self.logger.info(train_loss_output)
                if self.rank == 0:
                    self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
                self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step':epoch_idx}, head='train')           
            # eval

            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                    (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                if self.rank == 0:
                    self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                    for name, value in valid_result.items():
                        self.tensorboard.add_scalar(name.replace('@','_'), value, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step+=1
        
        #if self.rank == 0:
            #self._add_hparam_to_tensorboard(self.best_valid_score)            
        return self.best_valid_score, self.best_valid_result

    @torch.no_grad() 
    def _full_sort_batch_eval(self, batched_data):
        user, history_index, positive_u, positive_i = batched_data
        interaction = self.to_device(user)
               
        scores = self.model.module.predict(interaction ,self.item_feature)  #[eval_batch]    #[item_num, (feature_dim)] 
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        # self.logger.info(scores)
        # self.logger.info(scores.shape)
        # self.logger.info(history_index[0].shape)
        # self.logger.info(history_index[1].shape)
        # self.logger.info(scores)
        # self.logger.info(user)
        # self.logger.info(history_index[0])
        # self.logger.info(history_index[1])
        # self.logger.info(positive_u)
        # self.logger.info(positive_i)
        # import sys
        # sys.exit()

        if history_index is not None:
            scores[history_index] = -np.inf
        return scores, positive_u, positive_i
    
    @torch.no_grad()   
    def compute_item_feature(self,config,data):
        if self.use_modality:
            item_data = BatchDataset(config, data)
            item_loader = DataLoader(item_data, batch_size=100, num_workers=10, shuffle=False,pin_memory=True)
            self.item_feature = []
            
            with torch.no_grad():
                for idx, items in enumerate(item_loader):
                    items = items.to(self.device)
                    items = self.model.module.compute_item(items)
                    self.item_feature.append(items)
                if isinstance(items, tuple):
                    self.item_feature = torch.cat([x[0] for x in self.item_feature]), torch.cat([x[1] for x in self.item_feature])

                else:
                    self.item_feature = torch.cat(self.item_feature)  #[nitem, 64]
        else :
            with torch.no_grad():
                self.item_feature = self.model.module.compute_item_all() #[nitem, 64]

    def distributed_concat(self,tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        # truncate the dummy elements added by SequentialDistributedSampler
        return concat.sum()/ num_total_examples

   
    @torch.no_grad()    
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if not eval_data:
            return
        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file,map_location=torch.device('cpu'))
            self.model.module.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)
        
   
        self.model.eval()  
        eval_func = self._full_sort_batch_eval
                      
        self.tot_item_num = eval_data.dataset.dataload.item_num
        self.compute_item_feature(self.config, eval_data.dataset.dataload) 
        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=85,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )

        for batch_idx, batched_data in enumerate(iter_data):
            # self.logger.info(batched_data)
            # import sys
            # if batch_idx > 5:
            #     sys.exit()
            scores , positive_u, positive_i = eval_func(batched_data)

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, positive_u, positive_i)       
        num_total_examples = len(eval_data.sampler.dataset)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)

        metric_decimal_place = 5 if self.config['metric_decimal_place'] == None else self.config['metric_decimal_place']
        for k, v in result.items():
            result_cpu = self.distributed_concat(torch.tensor([v]).to(self.device),num_total_examples).cpu()
            #print(k,result_cpu)
            result[k] = round(result_cpu.item(),metric_decimal_place)
        self.wandblogger.log_eval_metrics(result, head='eval')

        return result

             
 