import json
import os
from datetime import datetime


def load_config(config_path):
    # remove comments starting with '//'
    json_str = ''
    with open(config_path, 'r') as json_file:
        for line in json_file:
            line = line.split('//')[0] + '\n'
            json_str += line

    args = json.loads(json_str)

    return args

def make_output_dirs(args):
    experiment_name = datetime.now().strftime('%y%m%d_%H%M%S') + "-" + args['name']
    args['name'] = experiment_name
    if args["opt"] == "train":
        args['path']['checkpoint_dir'] = os.path.join(args['path']['output_dir'], experiment_name,
                                                      args['path']['checkpoint_dir'])
    args['path']['log_dir'] = os.path.join(args['path']['output_dir'], experiment_name,
                                           args['path']['log_dir'])
    args['path']['vis_train_dir'] = os.path.join(args['path']['output_dir'], experiment_name,
                                                 args['path']['vis_train_dir'])
    args['path']['vis_val_sim_dir'] = os.path.join(args['path']['output_dir'], experiment_name,
                                                   args['path']['vis_val_sim_dir'])
    args['path']['vis_val_real_dir'] = os.path.join(args['path']['output_dir'], experiment_name,
                                                    args['path']['vis_val_real_dir'])

    if args["opt"] == "train":
        os.makedirs(args['path']['checkpoint_dir'], exist_ok=True)
        os.makedirs(args['path']['vis_train_dir'], exist_ok=True)
    os.makedirs(args['path']['log_dir'], exist_ok=True)
    os.makedirs(args['path']['vis_val_sim_dir'], exist_ok=True)
    os.makedirs(args['path']['vis_val_real_dir'], exist_ok=True)
