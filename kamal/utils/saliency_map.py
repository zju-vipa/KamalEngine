from numpy.lib.type_check import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.autograd import Variable
from PIL import Image
# from torch.tensor import Tensor


class inputGradient():
    def __init__(self, model, loss=False) -> None:
        self.model = model
        self.loss = loss

    def _getGradients(self, image: torch.Tensor, target_class=None):
        image.requires_grad = True
        outputs = self.model(image)

        if target_class is None:
            target_class = (outputs.data.max(1, keepdim=True)[1]).flatten()

        if self.loss:
            outputs = torch.log_softmax(outputs, 1)
            agg = F.nll_loss(outputs, target_class, reduction='sum')
        else:
            agg = -1. * F.nll_loss(outputs, target_class, reduction='sum')

        self.model.zero_grad()
        # Gradients w.r.t. input and features
        # outputs-Y  inputs-X  grad_outputsl-sum weights only_inputs-only return the gradients of inputs
        gradients = torch.autograd.grad(outputs=agg,
                                        inputs=image,
                                        only_inputs=True,
                                        retain_graph=True)[0]
        image.requires_grad = False
        # First element in the feature list is the image
        return gradients

    def saliency(self, image, target_class=None):
        self.model.eval()
        input_grad = self._getGradients(image, target_class=target_class)
        return [torch.abs(input_grad).sum(1, keepdim=True)]


class featureGrad():
    def __init__(self, model) -> None:
        self.model = model

    def __getGradients(self, image: torch.Tensor):
        image.requires_grad = True
        output, features = self.model(image,return_features=1)
        feature_gradients=[]
        for feature in features:
            agg = -torch.norm(feature.flatten())
            #agg = feature.flatten()
            self.model.zero_grad()
            gradients = torch.autograd.grad(outputs=agg,
                                            inputs=image,
                                            grad_outputs=torch.ones_like(agg),
                                            only_inputs=True,
                                            retain_graph=True)[0]
            gradients=gradients*image
            feature_gradients.append(gradients)
        image.requires_grad = False
        return feature_gradients

    def saliency(self,image):
        self.model.eval()
        input_grads=self.__getGradients(image)
        final_grads=[]
        for grad in input_grads:
            final_grads.append(torch.abs(grad).sum(1, keepdim=True))
        return final_grads

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos,module in enumerate(self.model.children()):
            if isinstance(module,nn.Linear):
                x=x.view(x.size(0),-1)
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def saliency(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        batch_size=input_image.shape[0]
        channel_num=input_image.shape[1]
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy(),axis=1)
        # Target for backprop
        one_hot_output = torch.FloatTensor(batch_size, model_output.size()[-1]).zero_()
        for i in range(batch_size):
            one_hot_output[i][target_class[i]] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()
        # Get convolution outputs
        target = conv_output.data.numpy()
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(2, 3))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones([batch_size,target.shape[2],target.shape[3]], dtype=np.float32)
        res_cam=[]
        # Multiply each weight with its conv output and then, sum
        for k in range(batch_size):
            for i, w in enumerate(weights[k]):
                cam[k] += w * target[k, i, :, :]
            cam[k] = np.maximum(cam[k], 0)
            cam[k] = (cam[k] - np.min(cam[k])) / (np.max(cam[k]) - np.min(cam[k]))  # Normalize between 0-1
            cam[k] = np.uint8(cam[k] * 255) 
            temp = cv2.resize(cam[k],(input_image.shape[2],input_image.shape[3]),interpolation=cv2.INTER_AREA)/255# Scale between 0-255 to visualize
            res_cam.append(temp)

        res_cam = torch.unsqueeze(torch.Tensor(np.array(res_cam)),1)

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return [res_cam]

def save_saliency_map(in_image, saliency_maps, i , filename):
    """ 
    Save saliency map on image.
    
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W) 
        filename: string with complete path and file extension
    """
    #print(type(image))
    origin_image=in_image.cpu().numpy().copy()
    origin_image=np.uint8(origin_image*255).transpose(1,2,0)
    origin_image=cv2.resize(origin_image,(224,224))

    save_list=[origin_image]
    
    for saliency_map in saliency_maps:
        image = in_image.cpu().numpy().copy()
        saliency_map = saliency_map[i].data.cpu().numpy()

        saliency_map = saliency_map - saliency_map.min()
        saliency_map = saliency_map / saliency_map.max()
        saliency_map = saliency_map.clip(0,1)

        saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
        saliency_map = cv2.resize(saliency_map, (224,224))

        image = np.uint8(image * 255).transpose(1,2,0)
        image = cv2.resize(image, (224, 224))

        # Apply JET colormap
        color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
        
        # Combine image with heatmap
        img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
        img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
        save_list.append(img_with_heatmap)

    imgs=np.concatenate(save_list,axis=1)
    cv2.imwrite(filename, np.uint8(255 * imgs))        


if __name__ == '__main__':
    pass
