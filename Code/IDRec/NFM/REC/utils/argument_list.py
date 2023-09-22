general_arguments = [
    'seed',
    'reproducibility',
    'state',
    'model',
    'data_path',
    'checkpoint_dir',
    'show_progress',
    'config_file',
    'save_data',
    'data_save_path',
    'log_wandb',
]

training_arguments = [
    'epochs', 'train_batch_size',
    'learner', 'learning_rate',
    'eval_step', 'stopping_step',
    'clip_grad_norm',
    'weight_decay',
    'loss_decimal_place',
]

evaluation_arguments = [
    'eval_args', 'repeatable',
    'metrics', 'topk', 'valid_metric', 'valid_metric_bigger',
    'eval_batch_size',
    'metric_decimal_place',
]

dataset_arguments = [
    'USER_ID_FIELD', 'ITEM_ID_FIELD','TIME_FIELD','MAX_ITEM_LIST_LENGTH'
]





