### Generative Adversarial Network (GAN) for Cranial Implant Generation ([generator](https://dl.dropboxusercontent.com/s/5tj6r8wgvxc4p8o/generator.txt?dl=0), [discriminator](https://dl.dropboxusercontent.com/s/30fhk8t34csp1m9/discriminator.txt?dl=0))
the GAN implementation is adapted from [@Fdevmsy/3D_shape_inpainting](https://github.com/Fdevmsy/3D_shape_inpainting).

![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/assets/gan_results.png)

| Network|Codes| Generator|discriminator|
| ------      | ------ | ------ | ------ |
| GAN | EncoderDecoderGAN3D.py |[generator.h5](https://files.icg.tugraz.at/f/9d5ee3d750294301b1c4/?dl=1)| [discriminator.h5](https://files.icg.tugraz.at/f/c83cf7be4d4246faa137/?dl=1)|
| AE | EncoderDecoder.py|[encoderdecoder.h5](https://files.icg.tugraz.at/f/9e5473d9d1ca4287bdf7/?dl=1)| - |
| AE (-n) | EncoderDecoder_patch.py|| - |

```python
python filename.py
```

### Boundary (Surface) Constrained Loss Function for Volumetric Shape Completion ([auto-encoder (AE)](https://dl.dropboxusercontent.com/s/5tj6r8wgvxc4p8o/generator.txt?dl=0))
the boundary loss implementation is adapted from [@LIVIAETS/boundary-loss](https://github.com/LIVIAETS/boundary-loss).


#### Euclidean Distance Transform (EDT) of a ground truth implant, viewed in axial, sagittal and coronal plane:
| axial| sagittal| coronal|3D|
| ------      | ------ | ------ |------ |
| ![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/assets/axial.gif) |![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/assets/sagittal.gif)|![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/assets/coronal.gif)|![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/assets/3DEDT.png)|
|![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/assets/snapshot0001.png)|![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/assets/snapshot0002.png)|![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/assets/snapshot0003.png)|![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/assets/3DIMP.png)|





| Loss function|Codes|Trained Model|
| ------      | ------ | ------ |
| [boundaryloss](https://www.sciencedirect.com/science/article/pii/S1361841520302152?via%3Dihub) |EncoderDecoder_boundaryloss.py | [boundaryloss.h5](https://files.icg.tugraz.at/f/774c9d3adca04dcebecf/?dl=1)|
| MSE | EncoderDecoder.py  |[encoderdecoder.h5](https://files.icg.tugraz.at/f/9e5473d9d1ca4287bdf7/?dl=1)|
| DICE | EncoderDecoder_diceloss.py |[diceloss.h5](https://files.icg.tugraz.at/f/2b455ed99bd442fbbeaf/?dl=1)|
| MSE+boundaryloss  |   ||
|DICE+boundaryloss  | EncoderDecoder_dice_boundary_loss.py <br /> EncoderDecoder_dice_boundary_loss_test.py |[dice_boundary_loss.h5](https://files.icg.tugraz.at/f/1a256d22dd3543809ec8/?dl=1)|

```python
python filename.py
```
> Note: for EncoderDecoder_boundaryloss.py, the EDT is computed during the data loading phase, which is compatible with numpy operations. For MSE+boundaryloss and DICE+boundaryloss, the EDT has to be computed within the tensorflow graph, which does not support numpy operations. tensor.numpy() can be used to convert a tensor to a normal python numpy array. To execute numpy arrays within graph, the eager execution mode has to be enabled, which is supported by Tensorflow 1.8 and above. For Tensorflow 2.X, the mode is enabled by default.      

Keras version of DICE Loss ([sources](https://github.com/keras-team/keras/issues/3611))
```python
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -100*dice_coef(y_true, y_pred)
```
Dice loss constrained by [boundary loss](https://github.com/LIVIAETS/boundary-loss) (keras version)
```python
def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def surface_loss_keras(y_true, y_pred):
    y_true = y_true.numpy()
    gt_dist_transform=np.array([calc_dist_map(y) for y in y_true]).astype(np.float32) 
    multipled = y_pred * gt_dist_transform
    return K.mean(multipled)


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_boundary_loss(y_true,y_pred):
    loss1=surface_loss_keras(y_true,y_pred)
    loss2=dice_coef_loss(y_true,y_pred)
    finalloss=loss1 + 100*loss2
    return finalloss
```
The convergence mode of a network when trained with the boundary loss (arrow indicate the change direction of the network's output):  

![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/assets/boundaryconvergence.png)

## Contact
Jianning Li (jianningli.me [at] gmail [dot] com)
