### Generative Adversarial Network (GAN) for Cranial Implant Generation
the GAN implementation is adapted from [@Fdevmsy/3D_shape_inpainting](https://github.com/Fdevmsy/3D_shape_inpainting).


| Codes|File| Generator|discriminator|
| ------      | ------ | ------ | ------ |
| GAN | EncoderDecoderGAN3D.py |[generator.h5](https://files.icg.tugraz.at/f/9d5ee3d750294301b1c4/?dl=1)| [discriminator.h5](https://files.icg.tugraz.at/f/c83cf7be4d4246faa137/?dl=1)|
| AE | EncoderDecoder.py|[encoderdecoder.h5](https://files.icg.tugraz.at/f/9e5473d9d1ca4287bdf7/?dl=1)| - |

```python
python filename.py
```

### Boundary (Surface) Constrained Loss Function for Volumetric Shape Completion
the boundary loss implementation is adapted from [@LIVIAETS/boundary-loss](https://github.com/LIVIAETS/boundary-loss).


#### Euclidean Distance Transform (EDT) of a ground truth implant, viewed in axial, sagittal and coronal plane:
| axial| sagittal| coronal|
| ------      | ------ | ------ |
| ![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/axial.gif) |![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/sagittal.gif)|![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/coronal.gif)|





| Codes|File|Trained Model|
| ------      | ------ | ------ |
| [boundaryloss](https://www.sciencedirect.com/science/article/pii/S1361841520302152?via%3Dihub) |EncoderDecoder_boundaryloss.py | [boundaryloss.h5](https://files.icg.tugraz.at/f/774c9d3adca04dcebecf/?dl=1)|
| MSE | EncoderDecoder.py  |[encoderdecoder.h5](https://files.icg.tugraz.at/f/9e5473d9d1ca4287bdf7/?dl=1)|
| DICE |  ||
| MSE+boundaryloss  |  ||
|DICE+boundaryloss  |  ||


Keras version of DICE Loss ([sources](https://github.com/keras-team/keras/issues/3611))
```python
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -100*dice_coef(y_true, y_pred)
```

