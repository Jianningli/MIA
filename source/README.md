### Generative Adversarial Network (GAN) for Cranial Implant Generation
the GAN implementation is adapted from [@Fdevmsy/3D_shape_inpainting](https://github.com/Fdevmsy/3D_shape_inpainting). All credits go to the original author(s). 


| Codes|File| Generator|discriminator|
| ------      | ------ | ------ | ------ |
| GAN | EncoderDecoderGAN3D.py |[generator.h5](https://files.icg.tugraz.at/f/9d5ee3d750294301b1c4/?dl=1)| [discriminator.h5](https://files.icg.tugraz.at/f/c83cf7be4d4246faa137/?dl=1)|
| AE | EncoderDecoder.py|[encoderdecoder.h5](https://files.icg.tugraz.at/f/9e5473d9d1ca4287bdf7/?dl=1)| - |



### Boundary (Surface) Constrained Loss Function for Volumetric Shape Completion
the boundary loss implementation is adapted from [@LIVIAETS/boundary-loss](https://github.com/LIVIAETS/boundary-loss). All credits go to the original author(s). 


#### Euclidean Distance Transform (EDT) of a ground truth implant, viewed in axial, sagittal and coronal plane:
| axial| sagittal| coronal|
| ------      | ------ | ------ |
| ![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/axial.gif) |![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/sagittal.gif)|![alt text](https://github.com/Jianningli/MIA/blob/add-license-1/source/coronal.gif)|





| Codes|File|Trained Model|
| ------      | ------ |
| boundaryloss [1] |EncoderDecoder_boundaryloss.py | [boundaryloss.h5](https://files.icg.tugraz.at/f/774c9d3adca04dcebecf/?dl=1)|
|  |  |
|  |  |
|  |  |
