
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
