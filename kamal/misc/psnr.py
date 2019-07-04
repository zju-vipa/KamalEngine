import torch


def psnr(img1, img2, size_average=True, data_range=255):
    N = img1.shape[0]
    mse = torch.mean(((img1-img2)**2).view(N, -1), dim=1)
    psnr = torch.clamp(torch.log10(data_range**2 / mse) * 10, 0.0, 99.99)
    if size_average == True:
        psnr = psnr.mean()

    return psnr
