
import argparse

from chemprop.features import get_available_features_generators


def get_args():
    parser = argparse.ArgumentParser(description='pytorch version of SGG')

    ''' Graph Settings '''
    parser.add_argument('--graph', type=bool, default=True)
    parser.add_argument('--gnn_atom_dim', type=int, default=0)
    parser.add_argument('--gnn_bond_dim', type=int, default=0)
    parser.add_argument('--gnn_activation', type=str, default='ReLU', help='ReLU, LeakyReLU, tanh...')
    parser.add_argument('--gnn_num_layers', type=int, default=5)
    parser.add_argument('--gnn_hidden_dim', type=int, default=256)

    ''' Sequence Settings '''
    parser.add_argument('--recons', type=bool, default=False)
    parser.add_argument('--seq_len', type=int, default=220)
    parser.add_argument('--sequence', type=bool, default=True)
    parser.add_argument('--seq_input_dim', type=int, default=64)
    parser.add_argument('--seq_num_heads', type=int, default=4)
    parser.add_argument('--seq_num_layers', type=int, default=4)
    parser.add_argument('--seq_hidden_dim', type=int, default=256)

    ''' Geometric Setting '''
    parser.add_argument('--geometry', type=bool, default=True)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--model_config", type=str, default="./model_configs/pretrain_gem.json")
    parser.add_argument('--geo_hidden_dim', type=int, default=256)
    parser.add_argument("--compound_encoder_config", type=str, default="./model_configs/geognn_l8.json")

    ''' Classifier '''
    parser.add_argument('--fusion', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=256)

    ''' Training'''
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bias', type=int, default=1)
    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_lr', type=float, default=2e-3, help='Maximum learning rate')
    parser.add_argument('--init_lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-3, help='Final learning rate')
    parser.add_argument('--cl_loss', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pro_num', type=int, default=1)
    parser.add_argument('--pool_type', type=str, default='attention')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cl_loss_num', type=int, default=0)
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')

    ''' Options '''
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='esol')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy'])
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--vocab_num', type=int, default=0)
    parser.add_argument('--task_type', type=str, default='reg', help='classification, reg')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined'])
    parser.add_argument('--folds_file', type=str, default=None, help='Optional file of fold labels')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='Use messages on atoms instead of messages on bonds')
    parser.add_argument('--max_data_size', type=int, help='Maximum number of data points to load')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--val_fold_index', type=int, default=None, help='Which fold to use as val for leave-one-out cross val')
    parser.add_argument('--test_fold_index', type=int, default=None, help='Which fold to use as test for leave-one-out cross val')
    parser.add_argument('--features_generator', type=str, nargs='*', choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')

    args = parser.parse_args()
    return args



