from kamal.layer_wise_ka import *
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
use_cuda = True
logfile = open("log.txt", 'w')


def main():

    teacher_num = 2
    model = "models/teacher1.pb"
    t1_param = "models/alexnet-part1-030.pkl"
    model2 = "models/teacher2.pb"
    t2_param = "models/alexnet-part2-081.pkl"

    namelist, nameinfo, modulelist = parser(model)
    namelist2, nameinfo2, modulelist2 = parser(model2)

    teacher1 = fetch_params(t1_param)
    teacher2 = fetch_params(t2_param)
    teachers = [teacher1, teacher2]
    teacherlist = []
    teacherparam = []
    for i in range(len(namelist)):
        tmp = [nameinfo[i], nameinfo2[i]]
        tmp1 = [modulelist[i], modulelist2[i]]
        teacherlist.append(tmp)
        teacherparam.append(tmp1)

    studentlist = [
        InfoStruct('Conv2d', 3, 72, (11, 11), 4, 2),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('MaxPool2d', None, None, 3, 2, None),
        InfoStruct('Conv2d', 72, 210, (5, 5), 1, 2),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('MaxPool2d', None, None, 3, 2, None),
        InfoStruct('Conv2d', 210, 420, (3, 3), 1, 1),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('Conv2d', 420, 320, (3, 3), 1, 1),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('Conv2d', 320, 320, (3, 3), 1, 1),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('MaxPool2d', None, None, 3, 2, None),
        InfoStruct('Dropout', None, None, None, None, None),
        InfoStruct('Linear', 11520, 4200, None, None, None),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('Dropout', None, None, None, None, None),
        InfoStruct('Linear', 4200, 4200, None, None, None),
        InfoStruct('ReLU', None, None, None, None, None),
        InfoStruct('Linear', 4200, 120, None, None, None)
    ]

    Net = KaNet(namelist, teacherlist, studentlist, teacherparam, 1e-4)
    lrs_autoencoder = [0.05, 0.025, 0.025, 0.01, 0.005, 0.005, 0.001, 0]
    Net.initialize_layers(lrs_autoencoder)
    train_loader, lentrain = get_dataset(
        "../../data/images_whole/train", batchsize=64)
    print("Train dataset size: {}".format(lentrain))
    Net.set_dataloader(train_loader)
    if torch_is_cuda():
        print("use_cuda!!!")

    Net.load_autoencoder("/temp_disk/yyl/save/autoencoder-29.pkl")
    # start_train_autoencoder(Net)
    lrs_layerwise = [0.5, 0.1, 0.02, 0.03, 0.05, 0.008, 0.005, 0.001]
    add_aux_layer(Net, lrs_layerwise)
    Net.load_partnet("save1/net-83.pkl")
    # Net.save_net("save2/net-62.pkl")
    start_train_layerwise(Net)

    #merge_distill_layers(Net,"save1/net-83.pkl", 'for-overall.pkl')
    # exit()
    # Net.load_jointnet("for-overall.pkl")
    Net.load_jointnet("/temp_disk/yyl/save/overall-295.pkl")
    start_train_overall(Net, teachers)


if __name__ == '__main__':
    main()
