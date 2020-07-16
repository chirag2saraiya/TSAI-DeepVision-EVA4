# Deploying to AWS


## Demo
### Input Image
![Input](assets/input.jpg)

### Screen shot
![demo](assets/demo.jpg)

### Demo URL
URL: [ https://7axvk27op0.execute-api.ap-south-1.amazonaws.com/dev/classify]( https://7axvk27op0.execute-api.ap-south-1.amazonaws.com/dev/classify)


## Instructions and Notes

Install Node and NPM

```bash
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
sudo apt-get install -y nodejs
```

Install serverless

```bash
sudo npm install -g serverless
```

Setup serverless

```bash
sudo chown -R $USER:$(id -gn $USER) /home/shadowleaf/.config
sls config credentials --provider aws --key **** --secret **** --overwrite
```

Use the template to create a new project

```bash
sls create --template aws-python3 --name mobilenet-pytorch-example
```


Install anaconda from https://docs.anaconda.com/anaconda/install/linux/

Create a new environment in Anaconda

```bash
conda create --name pytorch-env
conda activate pytorch-env
```

```bash
sls plugin install -n serverless-python-requirements
sls plugin install -n serverless-wsgi
```

Downloading Pretrained mobilenetv2 model

```python
>>> import torch
>>> model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
>>> traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
>>> traced_model.save('mobilenetv2.pt')
```

> IMPORTANT

Make sure to add Binary Media Types in Amazon API Gateway Settings

```txt
multipart/form-data
*/*
```

Deploy to SLS using

```bash
sls deploy
```



