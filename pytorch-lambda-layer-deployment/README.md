# aws-lambda-deployment-with-serverless

This repository contains example files on deploying pytorch based application to AWS Lambda using Serverless framework. For detailed explaination refer to this [blog](https://gaurav4664.medium.com/how-to-speed-up-aws-lambda-deployment-on-serverless-framework-by-leveraging-lambda-layers-623f7c742af4). 

Entire deployment procedure consists of deployment of static dependencies as Lambda layer and application code as Lambda function. It has been described below.

1. Create Deployment Package  
In this step we create deployment zip file containing python packages we want to deploy on our AWS Lambda Layer. Steps are as follows.

    - Create an empty Serverless package
          
          serverless create -template aws-python3 -path gp-pytorch-dependency-package

    - Install serverless-python-requirements plugin in our package
          
          cd gp-pytorch-dependency-package
          serverless plugin install -n serverless-python-requirements

    - Create a new [requirements.txt](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/pytorch-lambda-layer-deployment/gp-pytorch-dependency-package/requirements.txt) file. Add following python packages in it. 
    
          https://download.pytorch.org/whl/cpu/torch-1.5.1%2Bcpu-cp38-cp38-linux_x86_64.whl
          torchvision==0.6.1
          Pillow
          requests_toolbelt

    - Edit serverless.yml file to configure package creation as shown in [serverless.yml](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/pytorch-lambda-layer-deployment/gp-pytorch-dependency-package/serverless.yml). 
          
    - create package  
    
          serverless package

    - locate .requirements.zip file. Rename it to requirements.zip and zip it to create another zip file named ’torchLayerPackages.zip’  
    
          mv .requirements.zip requirements.zip
          zip torchLayerPackages.zip requirements.zip
          
    You can find files explained above [here.](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/pytorch-lambda-layer-deployment/gp-pytorch-dependency-package)

    Now our deployment package is ready.
    
2. Create Lambda Layer  
Now we will use Serverless to deploy our deployment package as Lambda Layer. Steps are given below.

    - Create an empty Serverless package. 
    
          serverless create -template aws-python3 -path gp-pytorch-lambda-layer
          
    - Copy torchLayerPackages.zip prepared in previous section to this package’s (gp-pytorch-lambda-layer) directory.
    
    - Edit serverless.yml file as shown [here](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/pytorch-lambda-layer-deployment/gp-pytorch-lambda-layer/serverless.yml). ‘artifact’ option in serverless.yml tells serverless path of the readymade deployment package.
    
    - Deploy layer  
    
          serverless deploy

    The layer we have deployed above has pytorch 1.5.1, torch vision 0.6.1, Pillow and its dependencies. We have made this layer global so anyone with its ARN can use it. How to use it in your application is shown in next step. 
    
    You can find files explained above [here.](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/pytorch-lambda-layer-deployment/gp-pytorch-lambda-layer)
    
3. Example Lambda function utilizing our Lambda Layer.

      Now we will make use of deployed Lambda layer in an example Lambda function.

      - Create an empty Serverless package  
        
            serverless create -template aws-python3 -path gp-pytorch-layer-test
            
      - Add a file unzip_requirements.py for unzipping requirements.zip file and adding into system path. You can find this file [here](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/pytorch-lambda-layer-deployment/gp-pytorch-layer-test/unzip_requirements.py)

      - Edit handler.py. You can find it [here](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/pytorch-lambda-layer-deployment/gp-pytorch-layer-test/handler.py)

        In the first few lines we are importing unzip_requirements.py. This will ensure that all the python packages from Lambda Layer have been imported and ready for use in our application.

        To test if packages are available, we will just import torch,torchvision and Pillow packages and print their version.

      - Edit serverless.yml. You can find it [here](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/pytorch-lambda-layer-deployment/gp-pytorch-layer-test/serverless.yml)
        
      - Deploy Lambda function
        
              serverless deploy
              
      You can find files explained above [here.](https://github.com/chirag2saraiya/TSAI-DeepVision-EVA4/blob/master/pytorch-lambda-layer-deployment/gp-pytorch-layer-test)              


