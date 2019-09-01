import argparse

from data_loader import Dataset
from trainer import Trainer
from tester import Evaluator
from RTN import RTN


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='name of dataset')
    parser.add_argument('-n', '--n-hidden', type=int, default=64,
                        help='size of embedding vector')
    parser.add_argument('-l', '--l-reg', type=float, default=0.01,
                        help='regularize strength')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='balance st and lt')
    parser.add_argument('--scale', type=float, default=5,
                        help='rating scale for prediction')
    parser.add_argument('--n-neg', type=int, default=9,
                        help='number of negative')
    parser.add_argument('-o', '--out-name', type=str, default='RTN.pt',
                        help='output filename')

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='name of optimizer (SGD, Adam, RMSprop, ...)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--clip', type=float, default=0,
                        help='gradient clipping value')

    parser.add_argument('--epochs', type=int, default=10000,
                        help='upper epochs limit')
    parser.add_argument('-b', '--batch-size', type=int, default=100,
                        help='batch size')
    parser.add_argument('-t', '--timestep', type=int, default=20,
                        help='timestep')
    # parser.add_argument('--unroll', type=int, default=-1,
    #                     help='unroll timestep for TBPTT')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience of training')

    parser.add_argument('--n-split', type=float, default=1,
                        help='split for train, val, test')
    parser.add_argument('--eval-step', type=int, default=5,
                        help='evaluation interval')
    parser.add_argument('--k', type=int, default=50,
                        help='K for topk evaluation')
    parser.add_argument('--n-candidates', type=int, default=-1,
                        help='number of candidates for evaluation (N<=0 will use all items)')
    parser.add_argument('--metric', type=str, default='ndcg',
                        help='eval metric: ndcg / recall')
    parser.add_argument('--not-sort', type=str2bool, nargs='?', const=True, default=False,
                        help='not sorting by timestamp')
    parser.add_argument('--repeatable', type=str2bool, nargs='?', const=True, default=False,
                        help='repeat interactions in datasets (e.g., Foursquare)')

    parser.add_argument('--device-id', type=int, default=None,
                        help='cuda device id')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers for dataset generator')
    parser.add_argument('--no-dump', type=str2bool, nargs='?', const=True, default=False,
                        help='Do not use dump')
    parser.add_argument('--verbose', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Dataset.load(args.dataset, n_split=args.n_split, sort=not args.not_sort, use_dump=not args.no_dump)

    model = RTN(Dataset.n_users, Dataset.n_items, args.n_hidden, l_reg=args.l_reg, scale=args.scale, alpha=args.alpha,
                n_neg=args.n_neg)
    print(model)

    _trainer = Trainer(model, out_name=args.out_name, metric=args.metric)
    _trainer.cuda(device_id=args.device_id)

    _trainer.set_optimizer(args.optimizer, clip=args.clip, lr=args.lr)
    _trainer.evaluator = Evaluator(model, k=args.k, users=3000, n_candidates=args.n_candidates,
                                   validate=True, timestep=args.timestep, repeatable=args.repeatable)
    _trainer.train(epochs=args.epochs, batch_size=args.batch_size, timestep=args.timestep,
                   eval_step=args.eval_step, patience=args.patience, verbose=args.verbose, num_workers=args.num_workers)

    model = RTN.load(_trainer.get_filedir())
    model.cuda()
    test_evaluator = Evaluator(model, k=args.k, n_candidates=args.n_candidates,
                                   validate=False, timestep=args.timestep, repeatable=args.repeatable)
    result = test_evaluator.run()
    _trainer.print_eval(result, title='Test result')
