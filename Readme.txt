The orignal VAE was driven from: https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb.

For more infromation about the MLR model visit: https://www.biorxiv.org/content/10.1101/2021.02.07.430171v3.full (code: https://github.com/Shekoo93/MLR)


1. Run the Training.py file

2. save the model in output 1

1. Run the modelRun.py

2. in the mVAE file, the function test_outputs() reproduce the images

3. labels_shape and labels_color variables are the numbers associated with the shape and the color of interest



Requirements:
The model was programmed in Python 3.7.6 . in a torch environment version 1.3.1. The imported packages are listed here: torch, numpy , torch.nn , torch.nn.functional, torch.optim , imageio, os, copy, matplotlib.pyplot , matplotlib.image , torchvision, datasets, transforms, utils, torch.autograd, Variable, torchvision.utils, save_image, sklearn, svm, sklearn.metrics, classification_report, confusion_matrix, tqdm, PIL, Image, ImageOps, ImageEnhance, version as PILLOW_VERSION, joblib, dump, load.
