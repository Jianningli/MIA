## Automatic Skull Defect Restoration and Cranial Implant Generation for Cranioplasty [[Bibtex](https://dl.dropboxusercontent.com/s/kdxyjbegma9iwv0/bibtex1.txt?dl=0)][[paper](https://doi.org/10.1016/j.media.2021.102171)]


## Appendix Information can be found [HERE](https://github.com/Jianningli/MIA/tree/add-license-1/source)

### A client-server based web application for automatic implant generation ([Project page](http://jianningli.me/autoCranialImp))

| Demo Site|Youtube Tutorial|
| ------      | ------ |
|[![Studierfenster](https://github.com/Jianningli/MIA/blob/add-license-1/images/website.PNG)](http://studierfenster.icg.tugraz.at/ "Studierfenster")  |  [![Skull Shape Reconstruction](https://github.com/Jianningli/MIA/blob/add-license-1/images/youtube.PNG)](https://www.youtube.com/watch?v=pt-jw8nXzgs&feature=youtu.be "Skull Shape Reconstruction")|



 ### patch-wise skull shape completion

The deep learning model is trained to perform the following actions:

```
1. scan the entire skull in a (3D) patch-wise manner
2. if patch contains no defected region:
3.     output the input
4. if patch contains the defected region:
5.     output the input and the missing part 
6. scan complete
7. patch stitched together
8. output: completed skull
```

![example](https://github.com/Jianningli/MIA/blob/add-license-1/images/patch-wise.gif)


### patch-wise direct implant generation
The missing part (i.e., the implant) can also be predicted directly without reconstructing the original skull, if the deep learning models are trained with the implants as the ground truth.

```
1. scan the entire skull in a (3D) patch-wise manner
2. if patch contains no defected region:
3.     output an empty (all 0) patch 
4. if patch contains the defected region:
5.     output only the missing part 
6. scan complete
7. patch stitched together
8. output:  implant
```
  
![example](https://github.com/Jianningli/MIA/blob/add-license-1/images/patch-wise-implant.gif)


### interpretation of learnt features for volumetric shape completion (Interpretable Deep Learning)
For better visualization, the 3D patch as well as the 3D feature maps are projected into a 2D plane.
We observe that the features learnt by a deep learning model for the volumetric shape completion task are interpretable and consistent/stable. 
The good interpretibility is large due to the <em>lightweight</em> nature (binary, sparse) of the skull data. 

![example](https://github.com/Jianningli/MIA/blob/add-license-1/images/features.png)


### Reference

If you find our repository useful or use the codes for your research, please use the following bibtex entry for reference of our work:

```
@article{li2021automatic,
  title={Automatic Skull Defect Restoration and Cranial Implant Generation for Cranioplasty},
  author={Li, Jianning and von Campe, Gord and Pepe, Antonio and Gsaxner, Christina and Wang, Enpeng and Chen, Xiaojun and Zefferer, Ulrike and T{\"o}dtling, Martin and Krall,     Marcell and Deutschmann, Hannes and others},
  journal={Medical Image Analysis},
  pages={102171},
  year={2021},
  publisher={Elsevier}
}
```


## Contact
Jianning Li (jianningli.me [at] gmail [dot] com)



