# Session1: MobileNets And Shufflenets

## Project Statement:  
Train [MobileNet-V2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) model on custom dataset (4 classes Flying Birds, Large Quadcopters, Small Quadcopters and Winged Drones) and deploy
it to AWS. 

## Demo
### Input Image
![Input](assets/300.jpg)

### Test Screenshot
![demo](assets/custom_mobilenet_aws_demo.png)

### Endpoint URL
URL: [https://v1agl77crf.execute-api.ap-south-1.amazonaws.com/dev/classify](https://v1agl77crf.execute-api.ap-south-1.amazonaws.com/dev/classify)


----------
## Under The Hood

### Code

### Resizing Strategy

### The Model
![Custom model](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/02-Transferlearning-Mobilenet/assets/custom_mobilenet.png)

### Training Analysis
![LossAccuracyCurve](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/02-Transferlearning-Mobilenet/assets/loss_Accuracy_curves.png)

### Missclassified Images  
- Flying Birds
![MissClassified](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/02-Transferlearning-Mobilenet/assets/misclassified_birds.png)


- Large QuadCopter
![MissClassified](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/02-Transferlearning-Mobilenet/assets/misclassified_large_quadcopters.png)


- Small QuadCopter
![MissClassified](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/02-Transferlearning-Mobilenet/assets/misclassified_small_quadcopters.png)


- Winged Drones
![MissClassified](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/02-Transferlearning-Mobilenet/assets/misclassified_winged_drones.png)

### Endpoint URL
URL: [https://v1agl77crf.execute-api.ap-south-1.amazonaws.com/dev/classify](https://v1agl77crf.execute-api.ap-south-1.amazonaws.com/dev/classify)

Used [Insomnia](https://insomnia.rest/download/) to query Endpoint as shown in Demo.

### References 
TSAI and https://towardsdatascience.com/scaling-machine-learning-from-zero-to-hero-d63796442526
