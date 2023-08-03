import os
import yaml
import random
import builtins
import argparse
import numpy as np
from shutil import copyfile
from espnet2.tasks.asr import ASRTask
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter

import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn
import torch.distributed as dist

from utils/datasets.lrs_dataset import LRSPretrainDataset, english_collate_fn
from utils/losses import LabelSmoothingCrossEntropy, DSALoss, ESALoss
from utils/metrics import ErrorCalculator

from models.encoder import Encoder
from models.decoder import Decoder
from models.video_net import Transformer

from kamal.distillation.c2kd.c2kd_trainer import C2KDTrainer
from kamal.distillation.c2kd.c2kd_task import C2KDTask

def init_env():
    torch.manual_seed(531)
    torch.cuda.manual_seed(531)
    np.random.seed(531)
    random.seed(531)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)


def init_tokenize():
    downloader = ModelDownloader()

    asr_config = downloader.download_and_unpack("kamo-naoyuki/wsj_transformer2")
    asr_model, asr_train_args = ASRTask.build_model_from_file(asr_config['asr_train_config'],
                                                              asr_config['asr_model_file'],
                                                              device='cpu')
    converter = TokenIDConverter(token_list=asr_model.token_list)
    tokenizer = build_tokenizer(token_type=asr_train_args.token_type)
    return converter, tokenizer, asr_model.sos, asr_model.eos, asr_model.ignore_id, len(asr_model.token_list)


def init_model(config, sos, eos, ignore_id, vocab_size):
    encoder = Encoder(input_dim=config['encoder_input_dim'], n_layers=config['encoder_layers'],
                      n_head=config['encoder_attention_heads'], d_model=config['d_model'],
                      ffn_dim=config['encoder_ffn_dim'],
                      dropout=config['dropout'], attention_dropout=config['attention_dropout'],
                      fc_dropout=config['fc_dropout'], max_positions=config['max_source_positions'])

    decoder = Decoder(sos_id=sos, eos_id=eos, ignore_id=ignore_id, vocab_size=vocab_size,
                      d_word_vec=config['d_word_vec'], n_layers=config['decoder_layers'],
                      n_head=config['decoder_attention_heads'], d_model=config['d_model'],
                      ffn_dim=config['decoder_ffn_dim'], dropout=config['dropout'],
                      attention_dropout=config['attention_dropout'], fc_dropout=config['fc_dropout'],
                      tgt_emb_prj_weight_sharing=config['tgt_emb_prj_weight_sharing'],
                      max_positions=config['max_target_positions'])

    model = Transformer(encoder=encoder, decoder=decoder)
    return model


def init_dataset(config):
    if config['dataset'] == 'LRS2' or config['dataset'] == 'LRS3':
        train_dataset = LRSPretrainDataset(visual_feature_dir=config['visual_feature_dir'],
                                           teacher_dir=config['teacher_dir'],
                                           filename=config['train_filename'], num_chs=config['num_chs'])
        val_dataset = LRSPretrainDataset(visual_feature_dir=config['visual_feature_dir'],
                                         teacher_dir=config['teacher_dir'],
                                         filename=config['val_filename'], num_chs=config['num_chs'])
        collate_fn = english_collate_fn

    return train_dataset, val_dataset, collate_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--param_file', type=str, default='./utils/12_6_2048.yaml')
    args = parser.parse_args()

    init_env()

    config = yaml.safe_load(open(args.param_file))
    torch.cuda.set_device(args.local_rank)
    if args.local_rank != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if 'MASTER_ADDR' in os.environ.keys():
        distributed = True
        env_dict = {key: os.environ[key] for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")}
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(backend='nccl')
    else:
        distributed = False

    converter, tokenizer, sos, eos, ignore_id, vocab_size = init_tokenize()
    model = init_model(config, sos, eos, ignore_id, vocab_size)
    model = model.cuda(args.local_rank)
    train_dataset, val_dataset, collate_fn = init_dataset(config)

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'],
                                               shuffle=train_shuffle, sampler=train_sampler, collate_fn=collate_fn,
                                               num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=collate_fn,
                                             num_workers=0, pin_memory=True)

    experiment_dir = config['experiment_dir']
    print('Expt:', experiment_dir)

    if args.local_rank == 0:
        # set up run path
        os.makedirs(experiment_dir, exist_ok=True)
        copyfile(args.param_file, '{}/hyper-parameters.yaml'.format(experiment_dir))

    print('batch num: ', len(train_loader))
    evaluate_step = len(train_loader) // config['eval_times_every_epoch']
    print_step = len(train_loader) // config['print_times_every_epoch']
    if print_step == 0:
        print_step = 1

    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                     patience=config['eval_times_every_epoch'] * config[
                                                         'lr_scheduler_wait'])
    error_calculator = ErrorCalculator(tokenizer=tokenizer, converter=converter,
                                       ignore_syms=['<blank>', '<unk>', '<NOISE>'],
                                       eos_id=eos, report_cer=True, report_wer=True)


    if config['cal_DSA']:
        DSA_criterion = DSALoss(ratio=config['alpha'])
    else:
        DSA_criterion = None

    if config['cal_ESA']:
        ESA_criterion = ESALoss(ratio=config['gamma'], seq_error_calculator=error_calculator, using_cer=True)
    else:
        ESA_criterion = None

    ce_criterion = LabelSmoothingCrossEntropy(ignore_id=ignore_id, smoothing=config['label_smoothing'])

    task = C2KDTask(name='C2KD_Task',
                                        ce_criterion=ce_criterion, 
                                        DSA_criterion=DSA_criterion, 
                                        ESA_criterion=ESA_criterion,
                                        error_calculator=error_calculator,
                                        attach_to=None)

    trainer = C2KDTrainer( 
        logger=kamal.utils.logger.get_logger('cifar10-sdb'), 
        tb_writer=SummaryWriter( log_dir='run/cifar10-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    trainer.setup(  model=model,
                    task=task,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    device=device
    )
    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)