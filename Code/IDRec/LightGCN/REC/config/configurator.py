import re
import os
import sys
import yaml
import torch
from logging import getLogger

from REC.evaluator import metric_types, smaller_metrics
from REC.utils import get_model, \
    general_arguments, training_arguments, evaluation_arguments, dataset_arguments, set_color


class Config(object):

    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):

        self._init_parameters_category()
        self.yaml_loader = self._build_yaml_loader()
        self.file_config_dict = self._load_config_files(config_file_list)
        self.variable_config_dict = self._load_variable_config_dict(config_dict)
        self.cmd_config_dict = self._load_cmd_line()
        self._merge_external_config_dict()

        self.model, self.model_class, self.dataset = self._get_model_and_dataset(model, dataset)
        self._load_internal_config_dict()
        self.final_config_dict = self._get_final_config_dict()
        self._set_default_parameters()
        


    def _init_parameters_category(self):
        self.parameters = dict()
        self.parameters['General'] = general_arguments
        self.parameters['Training'] = training_arguments
        self.parameters['Evaluation'] = evaluation_arguments
        self.parameters['Dataset'] = dataset_arguments

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(
                u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X
            ), list(u'-+0123456789.')
        )
        return loader

    def _convert_config_dict(self, config_dict):
        r"""This function convert the str parameters to their original type.

        """
        for key in config_dict:
            param = config_dict[key]
            if not isinstance(param, str):
                continue
            try:
                value = eval(param)
                if value is not None and not isinstance(value, (str, int, float, list, tuple, dict, bool, Enum)):
                    value = param
            except (NameError, SyntaxError, TypeError):
                if isinstance(param, str):
                    if param.lower() == "true":
                        value = True
                    elif param.lower() == "false":
                        value = False
                    else:
                        value = param
                else:
                    value = param
            config_dict[key] = value
        return config_dict

    def _load_config_files(self, file_list):
        file_config_dict = dict()
        if file_list:
            for file in file_list:
                with open(file, 'r', encoding='utf-8') as f:
                    file_config_dict.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config_dict

    def _load_variable_config_dict(self, config_dict):
        # HyperTuning may set the parameters such as mlp_hidden_size in NeuMF in the format of ['[]', '[]']
        # then config_dict will receive a str '[]', but indeed it's a list []
        # temporarily use _convert_config_dict to solve this problem
        return self._convert_config_dict(config_dict) if config_dict else dict()

    def _load_cmd_line(self):
        r""" Read parameters from command line and convert it to str.

        """
        cmd_config_dict = dict()
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                    raise SyntaxError("There are duplicate commend arg '%s' with different value." % arg)
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        #if len(unrecognized_args) > 0:
            #logger = getLogger()
            #logger.warning('command line args [{}] will not be used in RecBole'.format(' '.join(unrecognized_args)))
        cmd_config_dict = self._convert_config_dict(cmd_config_dict)
        return cmd_config_dict

    def _merge_external_config_dict(self):
        external_config_dict = dict()
        external_config_dict.update(self.file_config_dict)
        external_config_dict.update(self.variable_config_dict)
        external_config_dict.update(self.cmd_config_dict)
        self.external_config_dict = external_config_dict

    def _get_model_and_dataset(self, model, dataset):

        if model is None:
            try:
                model = self.external_config_dict['model']
            except KeyError:
                raise KeyError(
                    'model need to be specified in at least one of the these ways: '
                    '[model variable, config file, config dict, command line] '
                )
        if not isinstance(model, str):
            final_model_class = model
            final_model = model.__name__
        else:
            final_model = model
            final_model_class = get_model(final_model)

        if dataset is None:
            try:
                final_dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError(
                    'dataset need to be specified in at least one of the these ways: '
                    '[dataset variable, config file, config dict, command line] '
                )
        else:
            final_dataset = dataset
        
        return final_model, final_model_class, final_dataset

    def _update_internal_config_dict(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            config_dict = yaml.load(f.read(), Loader=self.yaml_loader)
            if config_dict is not None:
                self.internal_config_dict.update(config_dict)
        return config_dict

    def _load_internal_config_dict(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        overall_init_file = os.path.join(current_path, 'overall.yaml')

        self.internal_config_dict = dict()
       
        if os.path.isfile(overall_init_file):
            self._update_internal_config_dict(overall_init_file)
        
      

    def _get_final_config_dict(self):
        final_config_dict = dict()
        final_config_dict.update(self.internal_config_dict)
        final_config_dict.update(self.external_config_dict)
        return final_config_dict

    def _set_default_parameters(self):
        self.final_config_dict['dataset'] = self.dataset
        self.final_config_dict['model'] = self.model
        
        if hasattr(self.model_class, 'input_type'):
            self.final_config_dict['MODEL_INPUT_TYPE'] = self.model_class.input_type
        
        self.final_config_dict['data_path'] = os.path.join(self.final_config_dict['data_path'], self.dataset)
        metrics = self.final_config_dict['metrics']
        if isinstance(metrics, str):
            self.final_config_dict['metrics'] = [metrics]

        eval_type = set()
        for metric in self.final_config_dict['metrics']:
            if metric.lower() in metric_types:
                eval_type.add(metric_types[metric.lower()])
            else:
                raise NotImplementedError(f"There is no metric named '{metric}'")
        if len(eval_type) > 1:
            raise RuntimeError('Ranking metrics and value metrics can not be used at the same time.')
        self.final_config_dict['eval_type'] = eval_type.pop()

        # if self.final_config_dict['MODEL_TYPE'] == ModelType.SEQUENTIAL and not self.final_config_dict['repeatable']:
        #     raise ValueError('Sequential models currently only support repeatable recommendation, '
        #                      'please set `repeatable` as `True`.')

        valid_metric = self.final_config_dict['valid_metric'].split('@')[0]
        self.final_config_dict['valid_metric_bigger'] = False if valid_metric.lower() in smaller_metrics else True

        topk = self.final_config_dict['topk']
        if isinstance(topk, (int, list)):
            if isinstance(topk, int):
                topk = [topk]
            for k in topk:
                if k <= 0:
                    raise ValueError(
                        f'topk must be a positive integer or a list of positive integers, but get `{k}`'
                    )
            self.final_config_dict['topk'] = topk
        else:
            raise TypeError(f'The topk [{topk}] must be a integer, list')

        if 'additional_feat_suffix' in self.final_config_dict:
            ad_suf = self.final_config_dict['additional_feat_suffix']
            if isinstance(ad_suf, str):
                self.final_config_dict['additional_feat_suffix'] = [ad_suf]


    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getattr__(self, item):
        if 'final_config_dict' not in self.__dict__:
            raise AttributeError(f"'Config' object has no attribute 'final_config_dict'")
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = '\n'
        for category in self.parameters:
            args_info += set_color(category + ' Hyper Parameters:\n', 'pink')
            args_info += '\n'.join([(set_color("{}", 'cyan') + " =" + set_color(" {}", 'yellow')).format(arg, value)
                                    for arg, value in self.final_config_dict.items()
                                    if arg in self.parameters[category]])
            args_info += '\n\n'

        args_info += set_color('Other Hyper Parameters: \n', 'pink')
        args_info += '\n'.join([
            (set_color("{}", 'cyan') + " = " + set_color("{}", 'yellow')).format(arg, value)
            for arg, value in self.final_config_dict.items()
            if arg not in {
                _ for args in self.parameters.values() for _ in args
            }.union({'model', 'dataset', 'config_files'})
        ])
        args_info += '\n\n'
        return args_info

    def __repr__(self):
        return self.__str__()
