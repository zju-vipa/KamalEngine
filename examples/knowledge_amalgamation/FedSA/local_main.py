
import argparse
import copy
import os
from torch.utils.tensorboard import SummaryWriter
import torch,time
import os
import torch.nn as nn
import kamal

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from kamal import vision, engine, utils, amalgamation, metrics, callbacks


def main():
    parser = argparse.ArgumentParser()
    #global arguments
    parser.add_argument('--use_cpu',
                        action='store_true',
                        help="decide use cpu or not")
    parser.add_argument('--verbose',
                        action='store_true',
                        help="verbose output or not")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--resume',
                        action='store_true',
                        help="resume to train the model")
    parser.add_argument('--dataset',
                        type=str,
                        default="cifar10",
                        help="the dataset type for the algorithm")
    parser.add_argument('--bs',
                        type=int,
                        default=64,
                        help="amalgamation batch size")

    # local_training arguments
    parser.add_argument('--local_dataset',
                        type=str,
                        default='imagenet32',
                        help="the dataset type for the pretrained model")
    parser.add_argument('--num_clients',
                        type=int,
                        default=10,
                        help="number of users: K")
    parser.add_argument('--local_ep',
                        type=int,
                        default=10,
                        help="the number of local epochs: E1")
    parser.add_argument('--local_bs',
                        type=int,
                        default=20,
                        help="local batch size: B")
    parser.add_argument('--local_lr',
                        type=float,
                        default=0.01,
                        help="learning rate for local teacher")
    parser.add_argument('--local_momentum',
                        type=float,
                        default=0.5,
                        help="local SGD momentum (default: 0.5)")
    parser.add_argument('--teachers_dir',
                        type=str,
                        default='/home/by/ka/KamalEngine/examples/knowledge_amalgamation/FedSA/ckpt/teachers',
                        help='the output dir for ckpt of teachers')
    parser.add_argument('--finetune_epoch', type=int, default=1)

    parser.add_argument('--num_classes', type=int, default=5)

    # amalgamation arguments
    parser.add_argument('--center_lr',
                        type=float,
                        default=0.1,
                        help='learing rate for center KD')
    parser.add_argument('--server_lr',
                        type=float,
                        default=0.1,
                        help='learing rate for server KA')
    parser.add_argument('--probe_lr',
                        type=float,
                        default=0.1,
                        help="learning rate for probe data training")
    parser.add_argument('--ft_k',
                        type=float,
                        default=0.5,
                        help="k for translator of fitnets")
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help="SGD momentum (default: 0.5)")
    parser.add_argument('--model',
                        type=str,
                        default='resnet34',
                        help='model name')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help="rounds of training students")
    parser.add_argument('--local_ka_epoch',
                        type=int,
                        default=10,
                        help="rounds of local KA")
    parser.add_argument('--server_ka_epoch',
                        type=int,
                        default=1,
                        help="rounds of local KA")
    parser.add_argument('--alpha',
                        type=float,
                        default=0.5,
                        help="hyper-parameter for KD loss")
    parser.add_argument('--student_dir', type=str, default='./ckpt/')
    parser.add_argument('--student_ckpt',
                        type=str,
                        default='',
                        help="the path for the student checkpoint")

    #select models
    parser.add_argument(
        '--topk',
        type=int,
        default=2,
        help="select topk matched model according to attribute map")
    parser.add_argument('--topm',
                        type=int,
                        default=1,
                        help="select topm blocks")
    parser.add_argument(
        '--target_layer',
        type=int,
        default=6,
        help="the target layer for attribute map,resnet=6,resnt_t=5")
    
    parser.add_argument('--warmup',
                    default=20,
                    type=int,
                    metavar='N')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if args.local_dataset == 'imagenet32':
    dataset_train, dataset_test = utils.dataset_utils.getImageNet32(
        with_index=False)
    num_groups = [len(dataset_train.classes) // args.num_clients
                  ] * args.num_clients
    dict_users = utils.dataset_utils.split_dataset(dataset_train,
                                                   args.num_clients)
    test_dict_users = utils.dataset_utils.split_dataset(
        dataset_test, args.num_clients)

    num_classes = len(dataset_train.classes)
    base_model = vision.models.classification.resnet34(num_classes=num_classes,
                                                       pretrained=False)
    base_model.to(device)

    class_index = 0
    logger = utils.logger.get_logger('fedsa_imagenet32')
    tb_writer = SummaryWriter(log_dir='run/fedsaimagenet32-%s' %
                                  (time.asctime().replace(' ', '_')))
    for i in range(args.num_clients):
        model_path = os.path.join(args.teachers_dir, "model_{}.pth".format(i))

        local_data = utils.dataset_utils.DatasetSplit(dataset_train,
                                                      dict_users[i])
        local_dataloader = torch.utils.data.DataLoader(
            local_data, batch_size=args.local_bs, shuffle=True, num_workers=4)
        if "dataset_test" in dir():
            local_test_data = utils.dataset_utils.DatasetSplit(
                dataset_test, test_dict_users[i])
            local_test_dataloader = torch.utils.data.DataLoader(
                local_data,
                batch_size=args.local_bs,
                num_workers=4,
                shuffle=True)
        distribution_info = utils.dataset_utils.getDistributionInfo(local_data)
        # print(len(distribution_info)-distribution_info.count(0))

        local_net = copy.deepcopy(base_model)  #本地模型
        optimizer = torch.optim.Adam(local_net.parameters(), lr=args.local_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)
        best_acc = 0
        current_epoch = 0
        TOTAL_ITERS = len(local_dataloader) * 10
        if args.resume:
            save_dict = torch.load(model_path)
            best_acc = save_dict["best_acc"]
            current_epoch = save_dict["current_epoch"]
            local_net.load_state_dict(save_dict['weight'])

        print("\n====== start training {}th model ======".format(i))
        trainer = amalgamation.LocalTrainer(logger, tb_writer)
        trainer.setup(model=local_net,
                      dataloader=local_dataloader,
                      test_dataloader=local_test_dataloader,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      args=args,
                      device=device,
                      distribution_info=distribution_info,
                      class_index=class_index,
                      num_groups=num_groups,
                      count = i,
                      model_path = model_path)

        trainer.run(start_iter=0, max_iter=TOTAL_ITERS)
        #update current_epoch
        save_dict = torch.load(model_path)
        save_dict['current_epoch'] = args.local_ep

        class_index = class_index + num_groups[i]


if __name__ == "__main__":
    main()
