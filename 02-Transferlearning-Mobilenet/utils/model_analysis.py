import torch
import numpy as np
import matplotlib.pyplot as plt

def tensor_to_image(inp):
    img = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def plot_classwise_misclassified_Data(model,predict_data_loader,class_id,n_rows,n_cols,class_names):

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  #n_rows = 2
  #n_cols = 2
  done = False
  num_data = n_rows * n_cols
  misclassifiedImages = []
  trueLabels = []
  predLabels = []
  req_misclassified_indexes = []
  while len(misclassifiedImages) < num_data:  # iterate till required missclassified images found
    inputs, labels = next(iter(predict_data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    model.eval()
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    
    req_misclassified_indexes = [i for i in range(len(labels)) if ((preds[i] != labels[i]) and (labels[i]== class_id))]
    misclassifiedImages += inputs[req_misclassified_indexes] 
    trueLabels += labels[req_misclassified_indexes]
    predLabels += preds[req_misclassified_indexes]

  # Now we have missclassified images lets plot

  plt.style.use('dark_background')
  # Generate a rows x cols sized image grid 
  f, axarr = plt.subplots(n_rows,n_cols)


  for count in range(num_data) :
    axarr[count // n_cols][count % n_cols].imshow(tensor_to_image(misclassifiedImages[count]))
    axarr[count // n_cols][count % n_cols].set_title('Predicted:{}'.format(class_names[predLabels[count]]),fontsize=12)
    axarr[count // n_cols][count % n_cols].axis('off')
        
    
  f.subplots_adjust(hspace=0.2)    
  f.suptitle('List of Missclassified images for class:{}'.format(class_names[class_id]), fontsize=25)
  f.set_size_inches(n_cols*3,n_rows*3)

  return f

def plot_training_stat(history):
  f, ax = plt.subplots(1,2,figsize=(18,6))

  # plot losses
  for c in ['train_loss', 'valid_loss']:
      ax[0].plot(history[c], label=c)
      
  ax[0].legend()
  ax[0].set_xlabel('Epoch')
  ax[0].set_ylabel('Loss')
  _ = ax[0].set_title('Training and Validation Losses')


  #plot accuracy
  for c in ['train_acc', 'valid_acc']:
      ax[1].plot(100 * history[c], label=c)
      
  ax[1].legend()
  ax[1].set_xlabel('Epoch')
  ax[1].set_ylabel('Accuracy')
  _ = ax[1].set_title('Training and Validation Accuracy')
  
  return f


