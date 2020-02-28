import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import download_from_url, load_darknet_weights

model_urls = {
    'darknet19': 'https://pjreddie.com/media/files/darknet19.weights',
    'darknet19_448': 'https://pjreddie.com/media/files/darknet19_448.weights',
    'darknet53': 'https://pjreddie.com/media/files/darknet53.weights',
    'darknet53_448': 'https://pjreddie.com/media/files/darknet53_448.weights'
}


def conv3x3(in_planes, out_planes, padding=1, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     bias=False)


class BasicBlock(nn.Module):
    def __init__(self, planes, norm_layer=None, residual=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.residual = residual
        self.block = nn.Sequential(
            conv1x1(planes, planes // 2),
            norm_layer(planes // 2),
            nn.LeakyReLU(0.1, inplace=True),
            conv3x3(planes // 2, planes),
            norm_layer(planes),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.residual:
            out = out + identity
        return out

class DarkNet(nn.Module):
    def __init__(self, layers, num_classes=1000, pooling=False, residual=True):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.pooling = pooling
        self.residual = residual

        features = [
            conv3x3(3, self.inplanes),
            nn.BatchNorm2d(self.inplanes),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        features.extend(self._make_layer(64, layers[0]))
        features.extend(self._make_layer(128, layers[1]))
        features.extend(self._make_layer(256, layers[2]))
        features.extend(self._make_layer(512, layers[3]))
        features.extend(self._make_layer(1024, layers[4]))

        self.features = nn.Sequential(*features)
        self.conv = nn.Conv2d(1024,
                              num_classes,
                              kernel_size=(1, 1),
                              stride=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks):
        layers = []
        if self.pooling == True:
            layers.append(nn.MaxPool2d(2, 2))  # downsample with maxpooling
        layers.extend([
            conv3x3(self.inplanes, planes, stride=1 if self.pooling else 2),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(0.1, inplace=True),
        ])

        for _ in range(blocks):
            layers.append(BasicBlock(planes, residual=self.residual))
        self.inplanes = planes
        return layers

    def load_weights(self, weights_file, change):
        load_darknet_weights(self, weights_file)
        if change:
            new_order = [
                278, 212, 250, 193, 217, 147, 387, 285, 350, 283, 286, 353,
                334, 150, 249, 362, 246, 166, 218, 172, 177, 148, 357, 386,
                178, 202, 194, 271, 229, 290, 175, 163, 191, 276, 299, 197,
                380, 364, 339, 359, 251, 165, 157, 361, 179, 268, 233, 356,
                266, 264, 225, 349, 335, 375, 282, 204, 352, 272, 187, 256,
                294, 277, 174, 234, 351, 176, 280, 223, 154, 262, 203, 190,
                370, 298, 384, 292, 170, 342, 241, 340, 348, 245, 365, 253,
                288, 239, 153, 185, 158, 211, 192, 382, 224, 216, 284, 367,
                228, 160, 152, 376, 338, 270, 296, 366, 169, 265, 183, 345,
                199, 244, 381, 236, 195, 238, 240, 155, 221, 259, 181, 343,
                354, 369, 196, 231, 207, 184, 252, 232, 331, 242, 201, 162,
                255, 210, 371, 274, 372, 373, 209, 243, 222, 378, 254, 206,
                186, 205, 341, 261, 248, 215, 267, 189, 289, 214, 273, 198,
                333, 200, 279, 188, 161, 346, 295, 332, 347, 379, 344, 260,
                388, 180, 230, 257, 151, 281, 377, 208, 247, 363, 258, 164,
                168, 358, 336, 227, 368, 355, 237, 330, 171, 291, 219, 213,
                149, 385, 337, 220, 263, 156, 383, 159, 287, 275, 374, 173,
                269, 293, 167, 226, 297, 182, 235, 360, 105, 101, 102, 104,
                103, 106, 763, 879, 780, 805, 401, 310, 327, 117, 579, 620,
                949, 404, 895, 405, 417, 812, 554, 576, 814, 625, 472, 914,
                484, 871, 510, 628, 724, 403, 833, 913, 586, 847, 657, 450,
                537, 444, 671, 565, 705, 428, 791, 670, 561, 547, 820, 408,
                407, 436, 468, 511, 609, 627, 656, 661, 751, 817, 573, 575,
                665, 803, 555, 569, 717, 864, 867, 675, 734, 757, 829, 802,
                866, 660, 870, 880, 603, 612, 690, 431, 516, 520, 564, 453,
                495, 648, 493, 846, 553, 703, 423, 857, 559, 765, 831, 861,
                526, 736, 532, 548, 894, 948, 950, 951, 952, 953, 954, 955,
                956, 957, 988, 989, 998, 984, 987, 990, 687, 881, 494, 541,
                577, 641, 642, 822, 420, 486, 889, 594, 402, 546, 513, 566,
                875, 593, 684, 699, 432, 683, 776, 558, 985, 986, 972, 979,
                970, 980, 976, 977, 973, 975, 978, 974, 596, 499, 623, 726,
                740, 621, 587, 512, 473, 731, 784, 792, 730, 491, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 80, 81,
                82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
                98, 99, 100, 127, 128, 129, 130, 132, 131, 133, 134, 135, 137,
                138, 139, 140, 141, 142, 143, 136, 144, 145, 146, 2, 3, 4, 5,
                6, 389, 391, 0, 1, 390, 392, 393, 396, 397, 394, 395, 33, 34,
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 51, 49,
                50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                67, 68, 25, 26, 27, 28, 29, 30, 31, 32, 902, 908, 696, 589,
                691, 801, 632, 650, 782, 673, 545, 686, 828, 811, 827, 583,
                426, 769, 685, 778, 409, 530, 892, 604, 835, 704, 826, 531,
                823, 845, 635, 447, 745, 837, 633, 755, 456, 471, 413, 764,
                744, 508, 878, 517, 626, 398, 480, 798, 527, 590, 681, 916,
                595, 856, 742, 800, 886, 786, 613, 844, 600, 479, 694, 723,
                739, 571, 476, 843, 758, 753, 746, 592, 836, 714, 475, 807,
                761, 535, 464, 584, 616, 507, 695, 677, 772, 783, 676, 785,
                795, 470, 607, 818, 862, 678, 718, 872, 645, 674, 815, 69, 70,
                71, 72, 73, 74, 75, 76, 77, 78, 79, 126, 118, 119, 120, 121,
                122, 123, 124, 125, 300, 301, 302, 303, 304, 305, 306, 307,
                308, 309, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320,
                321, 322, 323, 324, 325, 326, 107, 108, 109, 110, 111, 112,
                113, 114, 115, 116, 328, 329, 606, 550, 651, 544, 766, 859,
                891, 882, 534, 760, 897, 521, 567, 909, 469, 505, 849, 813,
                406, 873, 706, 821, 839, 888, 425, 580, 698, 663, 624, 410,
                449, 497, 668, 832, 727, 762, 498, 598, 634, 506, 682, 863,
                483, 743, 582, 415, 424, 454, 467, 509, 788, 860, 865, 562,
                500, 915, 536, 458, 649, 421, 460, 525, 489, 716, 912, 825,
                581, 799, 877, 672, 781, 599, 729, 708, 437, 935, 945, 936,
                937, 938, 939, 940, 941, 942, 943, 944, 946, 947, 794, 608,
                478, 591, 774, 412, 771, 923, 679, 522, 568, 855, 697, 770,
                503, 492, 640, 662, 876, 868, 416, 931, 741, 614, 926, 901,
                615, 921, 816, 796, 440, 518, 455, 858, 643, 638, 712, 560,
                433, 850, 597, 737, 713, 887, 918, 574, 927, 834, 900, 552,
                501, 966, 542, 787, 496, 601, 922, 819, 452, 962, 429, 551,
                777, 838, 441, 996, 924, 619, 911, 958, 457, 636, 899, 463,
                533, 809, 969, 666, 869, 693, 488, 840, 659, 964, 907, 789,
                465, 540, 446, 474, 841, 738, 448, 588, 722, 709, 707, 925,
                411, 747, 414, 982, 439, 710, 462, 669, 399, 667, 735, 523,
                732, 810, 968, 752, 920, 749, 754, 961, 524, 652, 629, 793,
                664, 688, 658, 459, 930, 883, 653, 768, 700, 995, 549, 655,
                515, 874, 711, 435, 934, 991, 466, 721, 999, 481, 477, 618,
                994, 631, 585, 400, 538, 519, 903, 965, 720, 490, 854, 905,
                427, 896, 418, 430, 434, 514, 578, 904, 992, 487, 680, 422,
                637, 617, 556, 654, 692, 646, 733, 602, 808, 715, 756, 893,
                482, 917, 719, 919, 442, 563, 906, 890, 689, 775, 748, 451,
                443, 701, 797, 851, 842, 647, 967, 963, 461, 790, 910, 773,
                960, 981, 572, 993, 830, 898, 528, 804, 610, 779, 611, 728,
                759, 529, 419, 929, 885, 852, 570, 539, 630, 928, 932, 750,
                639, 848, 502, 605, 997, 983, 725, 644, 445, 806, 485, 622,
                853, 884, 438, 971, 933, 702, 557, 504, 767, 824, 959, 543
            ]
            conv_layers = [ layer for layer in self.modules() if isinstance( layer, nn.Conv2d)]
            last_conv_layer = conv_layers[-1]
            weight = last_conv_layer.weight.data
            new_weight = torch.zeros(weight.size(), dtype=weight.dtype)
            for i, idx in enumerate(new_order):
                new_weight[idx] = weight[i]
            last_conv_layer.weight.data.copy_( new_weight )
        

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x



def darknet19(pretrained=False, change=False, progress=True, **kwargs):
    model = DarkNet(layers=[0, 1, 1, 2, 2], pooling=True, residual=False)
    if pretrained:
        weights_file = download_from_url(model_urls['darknet19'],
                                         progress=progress)
        model.load_weights(weights_file, change)
    return model


def darknet19_448(pretrained=False, change=False, progress=True, **kwargs):
    model = DarkNet(layers=[0, 1, 1, 2, 2], pooling=True, residual=False)
    if pretrained:
        weights_file = download_from_url(model_urls['darknet19_448'],
                                         progress=progress)
        model.load_weights(weights_file, change)
    return model


def darknet53(pretrained=False, change=False, progress=True, **kwargs):
    model = DarkNet(layers=[1, 2, 8, 8, 4], pooling=False, residual=True)
    if pretrained:
        weights_file = download_from_url(model_urls['darknet53'],
                                         progress=progress)
        model.load_weights(weights_file, change)
    return model


def darknet53_448(pretrained=False, change=False, progress=True, **kwargs):
    model = DarkNet(layers=[1, 2, 8, 8, 4], pooling=False, residual=True)
    if pretrained:
        weights_file = download_from_url(model_urls['darknet53_448'],
                                         progress=progress)
        model.load_weights(weights_file, change)
    return model