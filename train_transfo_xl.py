# coding: utf-8
import argparse
import functools
import shutil
import time
import math
import os, sys
import itertools
from types import SimpleNamespace

import click
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from pytorch_pretrained_bert.tokenization_transfo_xl import get_lm_corpus
from pytorch_pretrained_bert.modeling_transfo_xl import TransfoXLLMHeadModel, TransfoXLConfig
from pytorch_pretrained_bert.training_transfo_xl_utilities import BalancedDataParallel

import logging


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')


def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'))


def update_dropout(m, dropout):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = dropout


def update_dropatt(m, dropatt):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = dropatt


###############################################################################
# Training code
###############################################################################

def evaluate(args, model, eval_iter):
    # Turn on evaluation mode which disables dropout.

    model.eval()

    instance = model
    if not hasattr(model, 'reset_length'):
        instance = model.module

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        instance.reset_length(args.eval_tgt_len,
                              args.ext_len + args.tgt_len - args.eval_tgt_len, args.mem_len)
    else:
        instance.reset_length(args.eval_tgt_len,
                              args.ext_len, args.mem_len + args.tgt_len - args.eval_tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.

    with torch.no_grad():
        mems = tuple()

        pbar = tqdm(enumerate(eval_iter), total=len(eval_iter), desc="Evaluating ... ")

        for i, (data, target, seq_len) in pbar:

            if 0 < args.max_eval_steps <= i:
                break

            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    # Switch back to the training mode
    instance.reset_length(args.tgt_len, args.ext_len, args.mem_len)

    model.train()

    return total_loss / total_len


def train(epoch, args, model, tr_iter, va_iter, optimizer, scheduler, logger, optimizer_sparse=None,
          scheduler_sparse=None):
    # Turn on training mode which enables dropout.

    train_step = 0
    moving_avg_train_loss = None
    alpha = 0.1
    best_val_loss = None

    model.train()
    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter

    pbar = tqdm(enumerate(train_iter), total=len(train_iter))

    for batch, (data, target, seq_len) in pbar:

        if train_step % args.eval_interval == 0:
            val_loss = evaluate(args, model, va_iter)

            log_str = 'Evaluation Epoch {:3d} | step {:>8d} | {:>6d} batches | lr {:.3g} | loss {:5.2f} | Perplexity {:5.2f}'.format(
                epoch,
                train_step,
                batch + 1,
                optimizer.param_groups[0]['lr'],
                val_loss,
                np.exp(val_loss)
            )

            logger(log_str)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if not args.debug:
                    with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                        torch.save(model.state_dict(), f)
                    with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)
                best_val_loss = val_loss

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)
                if args.sample_softmax > 0:
                    scheduler_sparse.step(val_loss)

        model.zero_grad()
        if args.batch_chunk > 1:
            data_chunks = torch.chunk(data, args.batch_chunk, 1)
            target_chunks = torch.chunk(target, args.batch_chunk, 1)
            for i in range(args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                ret = model(data_i, target_i, *mems[i])
                loss, mems[i] = ret[0], ret[1:]
                loss = loss.float().mean().type_as(loss) / args.batch_chunk
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
        else:
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

        if moving_avg_train_loss is None:
            moving_avg_train_loss = loss.float().item()
        else:
            moving_avg_train_loss = moving_avg_train_loss * (1 - alpha) + alpha * loss.float().item()

        log_str = 'Train Epoch {:3d} | step {:>8d} | {:>6d} batches | lr {:.3g} | loss {:5.2f} | Perplexity {:5.2f}'.format(
            epoch,
            train_step,
            batch + 1,
            optimizer.param_groups[0]['lr'],
            moving_avg_train_loss,
            np.exp(moving_avg_train_loss)
        )

        logger(log_str, print_=False)
        pbar.set_description(log_str)

        if args.fp16:
            optimizer.clip_master_grads(args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)


        if train_step == args.max_step:
            logger('-' * 100)
            logger('End of training')
            break


pass_config = click.make_pass_decorator(SimpleNamespace, ensure=True)


@click.group()
@click.option('--data', type=click.Path(dir_okay=True, exists=False), default='./data/freshly',
              help='location of the data corpus')
@click.option('--dataset', default='wt103', type=click.Choice(['wt103', 'lm1b', 'enwik8', 'text8']),
              help='dataset name')
@click.option('--lr', type=float, default=0.00025, help='initial learning rate (0.00025|5 for adam|sgd)')
@click.option('--mom', type=float, default=0.0, help='momentum for sgd')
@click.option('--scheduler', default='cosine', type=click.Choice(['cosine', 'inv_sqrt', 'dev_perf', 'constant']),
              help='lr scheduler to use.')
@click.option('--decay_rate', type=float, default=0.5, help='decay factor when ReduceLROnPlateau is used')
@click.option('--lr_min', type=float, default=0.0, help='minimum learning rate during annealing')
@click.option('--clip', type=float, default=0.25, help='gradient clipping')
@click.option('--clip_nonemb', is_flag=True, help='only clip the gradient of non-embedding params')
@click.option('--batch_size', type=int, default=60, help='batch size')
@click.option('--batch_chunk', type=int, default=1, help='split batch into chunks to save memory')
@click.option('--ext_len', type=int, default=0, help='length of the extended context')
@click.option('--not_tied', is_flag=True, default=False, help='do not tie the word embedding and softmax weights')
@click.option('--seed', type=int, default=1111, help='random seed')
@click.option('--cuda', is_flag=True, help='use CUDA')
@click.option('--adaptive', is_flag=True, default=True, help='use adaptive softmax')
@click.option('--pre_lnorm', is_flag=True, help='apply LayerNorm to the input instead of the output')
@click.option('--varlen', is_flag=True, help='use variable length')
@click.option('--multi_gpu', is_flag=True, help='use multiple GPU')
@click.option('--eval-interval', type=int, default=400, help='evaluation interval')
@click.option('--work_dir', default='./models/LM-TFM', type=str, help='experiment directory.')
@click.option('--restart', is_flag=True, default=False, help='restart training from the saved checkpoint')
@click.option('--restart_dir', type=str, default='', help='restart dir')
@click.option('--debug', is_flag=True, help='run in debug mode (do not create exp dir)')
@click.option('--same_length', is_flag=True, help='use the same attn length for all tokens')
@click.option('--attn_type', type=int, default=0, help='attention type. 0 for ours, 1 for Shaw et al, 2 for Vaswani '
                                                       'et al, 3 for Al Rfou et al.')
@click.option('--clamp_len', type=int, default=-1, help='use the same pos embeddings after clamp_len')
@click.option('--eta_min', type=float, default=0.0, help='min learning rate for cosine scheduler')
@click.option('--max_eval_steps', type=int, default=-1, help='max eval steps')
@click.option('--sample_softmax', type=int, default=-1, help='number of samples in sampled softmax')
@click.option('--patience', type=int, default=0, help='patience')
@click.option('--fp16', is_flag=True, help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
@click.option('--static-loss-scale', type=float, default=1, help='Static loss scale, positive power of 2 values can '
                                                                 'improve fp16 convergence.')
@click.option('--dynamic-loss-scale', is_flag=True, help='Use dynamic loss scaling.  If supplied, this argument '
                                                         'supersedes --static-loss-scale.')
@click.option('--init', default='normal', type=str, help='parameter initializer to use.')
@click.option('--emb_init', default='normal', type=str, help='parameter initializer to use.')
@click.option('--init_range', type=float, default=0.1, help='parameters initialized by U(-init_range, init_range)')
@click.option('--emb_init_range', type=float, default=0.01, help='parameters initialized by U(-init_range, init_range)')
@click.option('--init_std', type=float, default=0.02, help='parameters initialized by N(0, init_std)')
@click.option('--proj_init_std', type=float, default=0.01, help='parameters initialized by N(0, init_std)')
@click.option('--optim', default='adam', type=click.Choice(['adam', 'sgd', 'adagrad']), help='optimizer to use.')
@click.option('--freeze', is_flag=True, default=False)
@pass_config
def cli(ctx, **kwargs):
    ctx.__dict__.update(kwargs)


@cli.command()
@click.option('--n_layer', type=int, default=18, help='number of total layers')
@click.option('--n_head', type=int, default=16, help='number of heads')
@click.option('--d_head', type=int, default=64, help='head dimension')
@click.option('--d_embed', type=int, default=-1, help='embedding dimension')
@click.option('--d_model', type=int, default=1024, help='model dimension')
@click.option('--d_inner', type=int, default=4096, help='inner dimension in FF')
@click.option('--div_val', type=int, default=4, help='divident value for adapative input and softmax')
@click.option('--dropout', type=float, default=0.2, help='global dropout rate')
@click.option('--dropatt', type=float, default=0.2, help='attention probability dropout rate')
@click.option('--gpu0_bsz', type=int, default=0, help='batch size on gpu 0')
@click.option('--warmup_step', type=int, default=16000, help='upper epoch limit')
@click.option('--max_step', type=int, default=4000000, help='upper epoch limit')
@click.option('--tgt_len', type=int, default=384, help='number of tokens to predict')
@click.option('--mem_len', type=int, default=384, help='length of the retained previous heads')
@click.option('--eval_tgt_len', type=int, default=128, help='number of tokens to predict for evaluation')
@click.option('--pretrained_path', type=click.Path(exists=True), default='./models/transfo_xl_large.bin')
@click.option('--untie_r', is_flag=True, default=True)
@pass_config
def large(args, **kwargs):
    args.__dict__.update(**kwargs)
    main(args)


@cli.command()
@click.option('--n_layer', type=int, default=16, help='number of total layers')
@click.option('--n_head', type=int, default=10, help='number of heads')
@click.option('--d_head', type=int, default=41, help='head dimension')
@click.option('--d_embed', type=int, default=-1, help='embedding dimension')
@click.option('--d_model', type=int, default=410, help='model dimension')
@click.option('--d_inner', type=int, default=2100, help='inner dimension in FF')
@click.option('--div_val', type=int, default=1, help='divident value for adapative input and softmax')
@click.option('--dropout', type=float, default=0.1, help='global dropout rate')
@click.option('--dropatt', type=float, default=0.0, help='attention probability dropout rate')
@click.option('--gpu0_bsz', type=int, default=4, help='batch size on gpu 0')
@click.option('--warmup_step', type=int, default=0, help='upper epoch limit')
@click.option('--max_step', type=int, default=100000, help='upper epoch limit')
@click.option('--tgt_len', type=int, default=150, help='number of tokens to predict')
@click.option('--mem_len', type=int, default=150, help='length of the retained previous heads')
@click.option('--eval_tgt_len', type=int, default=150, help='number of tokens to predict for evaluation')
@click.option('--pretrained_path', type=click.Path(exists=False), default=None)
@click.option('--untie_r', is_flag=True, default=False)
@pass_config
def base(args, **kwargs):
    args.__dict__.update(**kwargs)
    main(args)


def main(args):
    args.tied = not args.not_tied

    if args.d_embed < 0:
        args.d_embed = args.d_model

    assert args.ext_len >= 0, 'extended context length must be non-negative'
    assert args.batch_size % args.batch_chunk == 0

    args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
    args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
    logger = create_exp_dir(args.work_dir,
                            scripts_to_save=['./train_transfo_xl.py',
                                             './pytorch_pretrained_bert/modeling_transfo_xl_utilities.py',
                                             './pytorch_pretrained_bert/modeling_transfo_xl.py',
                                             './pytorch_pretrained_bert/tokenization_transfo_xl.py',
                                             './pytorch_pretrained_bert/training_transfo_xl_utilities.py'
                                             ],
                            debug=args.debug)

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed_all(args.seed)

    # Validate `--fp16` option
    if args.fp16:
        if not args.cuda:
            print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
            args.fp16 = False
        else:
            try:
                from apex.fp16_utils import FP16_Optimizer
            except:
                print('WARNING: apex not installed, ignoring --fp16 option')
                args.fp16 = False

    device = torch.device('cuda' if args.cuda else 'cpu')

    # Load data
    ###############################################################################
    tokenizer = torch.load('./models/transfo_xl_vocab.bin')
    #from pytorch_pretrained_bert.tokenization_transfo_xl import TransfoXLCorpus

    #corpus = TransfoXLCorpus.from_pretrained("./models", "./models")

    corpus = get_lm_corpus(args.data, args.dataset, tokenizer=tokenizer)
    ntokens = len(corpus.vocab)
    args.n_token = ntokens

    eval_batch_size = 100
    tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len)
    va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
                                  device=device, ext_len=args.ext_len)
    te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
                                  device=device, ext_len=args.ext_len)

    # adaptive softmax / embedding
    cutoffs = []
    if args.adaptive:
        assert args.dataset in ['wt103', 'lm1b']
        if args.dataset == 'wt103':
            cutoffs = [20000, 40000, 200000]
        elif args.dataset == 'lm1b':
            cutoffs = [60000, 100000, 640000]

    if args.restart:
        with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
            model = torch.load(f)
        if not args.fp16:
            model = model.float()
        model.apply(lambda m: update_dropout(m, args.dropout))
        model.apply(lambda m: update_dropatt(m, args.dropatt))
    else:
        model_config = TransfoXLConfig(
            vocab_size_or_config_json_file=ntokens,
            cutoffs=cutoffs,
            d_model=args.d_model,
            d_embed=args.d_embed,
            n_head=args.n_head,
            d_head=args.d_head,
            d_inner=args.d_inner,
            div_val=args.div_val,
            pre_lnorm=args.pre_lnorm,
            n_layer=args.n_layer,
            tgt_len=args.tgt_len,
            ext_len=args.ext_len,
            mem_len=args.mem_len,
            clamp_len=args.clamp_len,
            same_length=args.same_length,
            proj_share_all_but_first=True,
            attn_type=args.attn_type,
            sample_softmax=args.sample_softmax,
            adaptive=args.adaptive,
            tie_weight=args.tied,
            dropout=args.dropout,
            dropatt=args.dropatt,
            untie_r=args.untie_r,
            init="normal",
            init_range=0.01,
            proj_init_std=0.01,
            init_std=0.02)

        model = TransfoXLLMHeadModel(model_config)

    if args.pretrained_path is not None:
        model.load_pretrained_from_path(args.pretrained_path)

    if args.freeze:
        for n, w in model.named_parameters():
            if 'crit' in n or 'word_emb' in n:
                w.requires_grad = True
            else:
                w.requires_grad = False

    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = sum([p.nelement() for p in model.transformer.layers.parameters()])
    args.n_trainable_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])

    if args.fp16:
        model = model.half()

    if args.multi_gpu:
        model = model.to(device)
        if args.gpu0_bsz >= 0:
            para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                              model, dim=1).to(device)
        else:
            para_model = nn.DataParallel(model, dim=1).to(device)
    else:
        para_model = model.to(device)

    # optimizer
    optimizer_sparse = None
    if args.optim.lower() == 'sgd':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
            optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.mom)
    elif args.optim.lower() == 'adam':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
            optimizer = optim.Adam(dense_params, lr=args.lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    # scheduler
    scheduler_sparse = None
    if args.scheduler == 'cosine':
        # here we do not set eta_min to lr_min to be backward compatible
        # because in previous versions eta_min is default to 0
        # rather than the default value of lr_min 1e-6
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         args.max_step, eta_min=args.eta_min)  # should use eta_min arg
        if args.sample_softmax > 0:
            scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
                                                                    args.max_step,
                                                                    eta_min=args.eta_min)  # should use eta_min arg
    elif args.scheduler == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and args.warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > args.warmup_step \
                    else step / (args.warmup_step ** 1.5)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == 'dev_perf':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=args.decay_rate, patience=args.patience,
                                                         min_lr=args.lr_min)
        if args.sample_softmax > 0:
            scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sparse,
                                                                    factor=args.decay_rate, patience=args.patience,
                                                                    min_lr=args.lr_min)
    elif args.scheduler == 'constant':
        pass

    if args.cuda and args.fp16:
        # If args.dynamic_loss_scale is False, static_loss_scale will be used.
        # If args.dynamic_loss_scale is True, it will take precedence over static_loss_scale.
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={'init_scale': 2 ** 16})

    if args.restart:
        if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
            with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
                opt_state_dict = torch.load(f)
                optimizer.load_state_dict(opt_state_dict)
        else:
            print('Optimizer was not saved. Start from scratch.')

    logger('=' * 100)
    for k, v in args.__dict__.items():
        logger('    - {} : {}'.format(k, v))
    logger('=' * 100)
    logger('#params = {}'.format(args.n_all_param))
    logger('#non emb params = {}'.format(args.n_nonemb_param))
    logger('#trainable params = {}'.format(args.n_trainable_param))

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in itertools.count(start=1):
            train(epoch, args, para_model, tr_iter, va_iter, optimizer, scheduler, logger=logger,
                  optimizer_sparse=optimizer_sparse,
                  scheduler_sparse=scheduler_sparse)

    except KeyboardInterrupt:
        logger('-' * 100)
        logger('Exiting from training early')
    # Load the best saved model.
    with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    para_model = model.to(device)
    # Run on test data.
    test_loss = evaluate(args, para_model, te_iter)
    logger('=' * 100)
    if args.dataset in ['enwik8', 'text8']:
        logger('| End of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
            test_loss, test_loss / math.log(2)))
    else:
        logger('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
            test_loss, np.exp(test_loss)))
    logger('=' * 100)


if __name__ == '__main__':
    cli()
