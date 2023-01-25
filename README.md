### UDepth: Fast Monocular Depth Estimation for Visually-guided Underwater Robots

<img src=/data/udepth.gif width=53% /> <img src=/data/udepth.jpeg width=45.5% /> 

### Pointers
- Preprint: https://arxiv.org/pdf/2209.12358.pdf
- Video demonstration: https://youtu.be/lNK90c5ofVQ

<img src=/data/RMI_space.jpeg width=55% /> <img src=/data/RMI_next.jpeg width=38% />

### Model and test scripts
- The UDepth model architecture is in [model/udepth.py](model/udepth.py)
- The saved models are in [saved_model](saved_model/)
- UDepth inference on [USOD10K](https://github.com/LinHong-HIT/USOD10K) dataset: use [test_usod10k.py](test_usod10k.py) 
- UDepth inference on other images: use [inference.py](inference.py) 
- Domain projection inference on USOD10K dataset: use [test_usod10k_proj.py](test_usod10k_proj.py) 
- Domain projection inference on other images: use [inference_proj.py](inference_proj.py) 


#### Bibliography entry:

    @inproceedings{yu2022udepth,    
    author={Yu, Boxiao and Wu, Jiayi and Islam, Md Jahidul},    
    title={UDepth: Fast Monocular Depth Estimation for Visually-guided Underwater Robots},    
    booktitle={Accepted at the IEEE International Conference on Robotics and Automation (ICRA)},    
    year={2023},    
    organization={IEEE}
    }


#### Acknowledgements
- https://github.com/shariqfarooq123/AdaBins
- https://github.com/wuzhe71/CPD
