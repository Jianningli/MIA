

* ## patch-wise skull completion
  the deep learning model scans the entire skull in a (3D) patch-wise manner. When the 3D patch contains no defected region, the model 
  simply reconstruct/reproduce the input patch. When the patch contains the defected region (e.g., the middle four patches), the model reproduces the input patch while at the same time filling the hole. The output is the same skull with the hole filled.    

![example](https://github.com/li-jianning/patch-based-skull-completion/blob/master/images/patch-wise.gif)

* ## patch-wise direct implant generation
  the missing part (i.e., the implant) can also be predicted directly without reconstructing the original skull. Unlike the skull shape  completion which predicts the complete skull first and then subtract the input to obtain the implant, directly implant generation allows prediction of the implant directly given a defective skull. Similarly, the deep learning model (trained differently from the shape completion model) scans the entire skull in a (3D) patch-wise manner. When the 3D patch contains no defected region, the model output zero (the background value). When the patch contains the defected region (e.g., the middle four patches), the model reconstructs only the missing region of the input patch. The output is the implant. 
![example](https://github.com/li-jianning/patch-based-skull-completion/blob/master/images/patch-wise-implant.gif)
