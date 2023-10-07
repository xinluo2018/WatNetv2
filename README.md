# WatNetv2
An improved version of the [WatNet](https://github.com/xinluo2018/WatNet). Specifically,  
1. The WatNetv2 is not applicable for Sentinel-2 image but also applicable for the Landsat series (Landsat-5, 7, 8 and 9) image.   
2. The WatNetv2 achieved high surface water mapping accuracy by compared with WatNet. 


To do:   
1. try the data augmentation by using torch module.
2. try adaptive multiscale network. specifically, if the fine-scale scene could reach to the accurate classification, the coarser-scale scene is used to classificaion.
3. modify the parallel_loader.py, make multiple scenes to multiple patches, rather than one scene to multiple patches. 




