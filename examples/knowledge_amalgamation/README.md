
## Comprehensive Classification (StanfordCars + FGVCAircraft)

### 2 ResNet18 => 1 ResNet18
|                |  Teacher  |  Scratch    |   KD     |  Layerwise KA | Common Feature | 
| :----:         |  :----:   |    :----:   | :----:   |    :----:     |  :----:        |
| Car            |   0.750   |   0.747     |  0.766   |     0.738     |    **0.773**   |
| Aircraft       |   0.699   |   0.688     |  0.710   |     0.707     |    **0.720**   |

### 4 ResNet18 => 1 ResNet18

|                |  Teacher  |  Scratch    |   KD     |  Layerwise KA | Common Feature | 
| :----:         |  :----:   |    :----:   | :----:   |    :----:     |  :----:        |
| Car            |   0.750   |   0.747     |  -   |     -     |    **0.737**   |
| Aircraft       |   0.699   |   0.634     |  -   |     -     |    **0.670**   |
| Dogs           |   0.644   |   0.545     |  -   |     -     |    **0.602**   |
| CUB            |   0.550   |   0.545     |  -   |     -     |    **0.590**   |

<img src="run/car-acc.png" width = "70%" alt="icon"/>  
<img src="run/aircraft-acc.png" width = "70%" alt="icon"/>  


## Joint Scene Parsing (Segmentation + Monocular Depth)

|                       |  Teacher  |  Scratch (Multi-Tasking)    |   KD     |  Task Branching |
| :----:                |  :----:   |    :----:                   | :----:   |    :----:     |
| Segmentation (mIoU)   |   0.519   |    0.528                    |  0.541   |    0.538        |
| Depth (RMSE)          |   0.689   |    0.657                    |  0.652   |    0.664        |

|                |  LR Finder  |  No Finder    | 
|      :----:    |    :----:   | :----:        | 
| lr             |   0.022     |    0.01       | 
| Acc           |             |               |