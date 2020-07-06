from .stream_metrics import Metric, MetricCompose
from .accuracy import Accuracy, TopkAccuracy
from .confusion_matrix import ConfusionMatrix, IoU, mIoU
from .regression import *

class StandardTaskMetrics(object):
    @staticmethod
    def classification():
        return MetricCompose(
            metric_dict={'acc': Accuracy()},
            primary_metric='acc'
        )

    @staticmethod
    def regression():
        return MetricCompose(
            metric_dict={'mse': MeanSquaredError()},
            primary_metric='mse'
        )

    @staticmethod
    def segmentation(num_classes, ignore_idx=255):
        confusion_matrix = ConfusionMatrix(num_classes=num_classes, ignore_idx=ignore_idx)
        return MetricCompose(
            metric_dict={'acc': Accuracy(), 'confusion_matrix': confusion_matrix , 
                         'mIoU': mIoU(confusion_matrix)},
            primary_metric='mIoU'
        )

    @staticmethod
    def monocular_depth():
        return MetricCompose(
            metric_dict={
                'rmse': RootMeanSquaredError(),
                'rmse_log': RootMeanSquaredError( output_target_transform=lambda x,y: ( (x+1e-8).log(), (y+1e-8).log() ) ),
                'rmse_scale_inv': ScaleInveriantMeanSquaredError(),
                'abs rel': AbsoluteRelativeDifference(),
                'sq rel': SquaredRelativeDifference(), 
                'percents within thresholds': Threshold( thresholds=[1.25, 1.25**2, 1.25**3] )
            },
            primary_metric='rmse'
        )

    