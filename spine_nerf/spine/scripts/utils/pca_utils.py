import torch

def sobel_gradients(img):
    # lazy import
    import torch.nn.functional as TF
    
    # Sobel kernels
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(img.device)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).to(img.device)

    # perform 2D convolution
    grad_x = TF.conv2d(img, sobel_x, padding=1)
    grad_y = TF.conv2d(img, sobel_y, padding=1)
    
    return torch.concat((grad_x, grad_y), dim=1)