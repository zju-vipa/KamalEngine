import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224 as create_student_model
from vit_model import vit_base_patch16_224 as create_teacher_model
from utils import read_split_data, train_one_epoch_of_KA, evaluate

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./animals_student") is False:
        os.makedirs("animals_student")

    tb_writer = SummaryWriter()

    teacher1 = create_teacher_model(num_classes=args.num_classes_of_teacher1).to(device)   # 还要载入用数据集训练好的weight文件
    teacher2 = create_teacher_model(num_classes=args.num_classes_of_teacher2).to(device)   # 还要载入用数据集训练好的weight文件
    student = create_student_model(num_classes=args.num_classes_of_teacher1+args.num_classes_of_teacher2, is_student=True).to(device)    # 要训练的模型
    # print(student.modules())
    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)
    #     # 删除不重要的权重
    #     del_keys = ['head.weight', 'head.bias']
    #     for k in del_keys:
    #         del weights_dict[k]
    #     print(teacher1.load_state_dict(weights_dict, strict=False))
    #     print(teacher2.load_state_dict(weights_dict, strict=False))

    # train_image_path_of_teacher1, train_image_label_of_teacher1, \
    # val_img_path_of_teacher1, val_image_label_of_teacher1 = read_split_data(args.data_path_of_teacher1)

    # train_image_path_of_teacher2, train_image_label_of_teacher2, \
    # val_img_path_of_teacher2, val_image_label_of_teacher2 = read_split_data(args.data_path_of_teacher2)

    teacher1.load_state_dict(torch.load(args.weights_of_teacher1))
    teacher2.load_state_dict(torch.load(args.weights_of_teacher2))
    # student.load_state_dict(torch.load(args.weights_of_student))
    train_image_path_of_student, train_image_label_of_student, \
    val_img_path_of_student, val_image_label_of_student = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # train_dataset_of_teacher1 = MyDataSet(images_path=train_image_path_of_teacher1, images_class=train_image_label_of_teacher1, transform=data_transform["train"])
    # train_dataset_of_teacher2 = MyDataSet(images_path=train_image_path_of_teacher2, images_class=train_image_label_of_teacher2, transform=data_transform["train"])
    train_dataset = MyDataSet(images_path=train_image_path_of_student, images_class=train_image_label_of_student, transform=data_transform["train"])
    # val_dataset_of_teacher1 = MyDataSet(images_path=val_img_path_of_teacher1, images_class=val_image_label_of_teacher1, transform=data_transform["val"])
    # val_dataset_of_teacher2 = MyDataSet(images_path=val_img_path_of_teacher2, images_class=val_image_label_of_teacher2, transform=data_transform["val"])
    val_dataset = MyDataSet(images_path=val_img_path_of_student, images_class=val_image_label_of_student, transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                         pin_memory=True, num_workers=nw,
                                                         collate_fn=val_dataset.collate_fn)
    # val_loader_of_teacher2 = torch.utils.data.Dataloader(val_dataset_of_teacher2, batch_size=batch_size, shuffle=False,
    #                                                      pin_memory=True, num_workers=nw,
    #                                                      collate_fn=val_dataset_of_teacher2.collate_fn)

    # pg = [p for p in student.parameters() if p.requires_grad]
    pg = [p for p in student.parameters()]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    block_params = []
    for block in student.blocks:
        block_params.extend(list(block.parameters()))
    amal_optimizer = torch.optim.SGD(block_params, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch_of_KA(model=student, teacher1=teacher1, teacher2=teacher2, optimizer=optimizer, amal_optimizer=amal_optimizer, data_loader=train_loader, device=device, epoch=epoch)
        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=student, data_loader=val_loader, device=device, epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if epoch % 5 == 0:
            torch.save(student.state_dict(), "./animals_student/model-{}.pth".format(epoch))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes_of_teacher1', type=int, default=45)
    parser.add_argument('--num_classes_of_teacher2', type=int, default=45)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="../animals-90")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights_of_teacher1', type=str, default='./animals_teacher1/model-198.pth',
                        help='initial weights path')
    parser.add_argument('--weights_of_teacher2', type=str, default='./animals_teacher2/model-198.pth',
                        help='initial weights path')
    # parser.add_argument('--weights_of_student', type=str, default='./weights_of_student_Stanford_Dogs/model-41.pth',
    #                      help='initial weights path')
    # # 是否冻结权重
    # parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
