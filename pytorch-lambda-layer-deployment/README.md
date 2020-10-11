## Lamda layer deployment of pytorch,torchvision and Pillow  

### Steps  
1. Follow the usual procedure for pytorch based function deployment to AWS lambda using Serverless. [Ref](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/tree/master/01-Mobilenet-Pytorch-Example)
2. Instead of deployment using **"serverless deploy"** command just use **"serverless package"** to just create a deployment package **".requirements.zip"**.
3. Rename **".requirements.zip"** as **"requirements.zip"** and put this under another zip file named **"torchLayerPackages.zip"**
4. Create new serverless service package for lambda layer deployment.
5. In the **"serverless.yml"** add a segment for layer deployment with artifact as **"torchLayerPackages.zip"** as shown in [serverless.yml](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/pytorch-lambda-layer-deployment/gp-pytorch-lambda-layer/serverless.yml). Doing this indicates serverless to deploy the given artifact .zip file as it is to lambda layer.
6. Execute **"serverless deploy"**. The zip file **"torchLayerPackages.zip"** will be unzipped while creating the lambda layer and its contents i.e. **"requirements.zip"** will be deployed as part of the lambda layer.
7. Now whenever the above deployed layer is used for any lambda function, layer contents i.e. **"requirements.zip"** will be copied into **/opt/** directory of that lambda function.
8. Next it is important to unzip the **"requirements.zip"** file and add the corresponding directory to system path so that python can see the layer packages. To do this we write a [**"unzip_requirements.py"**](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/pytorch-lambda-layer-deployment/gp-pytorch-layer-test/unzip_requirements.py) similar to one found in serverless deployments with requirements plugin.
9. Now for any lambda function deployments using our layer must execute the file **"unzip_requirements.py"** at the beginning. 
10. One simple test lambda deployment can be found [here](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/tree/master/pytorch-lambda-layer-deployment/gp-pytorch-layer-test).
