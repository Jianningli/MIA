

* ## patch-wise skull completion
  the deep learning model scans the entire skull in a (3D) patch-wise manner.  When the 3D patch contains no defected region, the model 
  simply reconstruct/reproduce the input patch. When the patch contains the defected region (e.g., the middle four patches), the model reproduces the input patch while at the same time filling the hole. The output is    

![example](https://github.com/li-jianning/patch-based-skull-completion/blob/master/images/patch-wise.gif)

* ## patch-wise direct implant generation
  the deep learning model scans the entire skull in a (3D) patch-wise manner. 
![example](https://github.com/li-jianning/patch-based-skull-completion/blob/master/images/patch-wise-implant.gif)
