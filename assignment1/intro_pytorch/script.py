import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
import os
from MNIST_dataloader import create_dataloaders

doc_dir = os.getcwd()
data_dir = doc_dir + '/data'
fig_dir = doc_dir + '/figures/'

def relu(input):
    return torch.from_numpy(np.maximum(input.cpu().detach().numpy(),0))

class ReLU(nn.Module):
    def __init__(self):
        super().__init__() # Initialising the base class

    def forward(self, input):
        return relu(input) # Applying the custom ReLU function

class Autoencoder(nn.Module):
    def __init__(self, use_activation=False):
        super().__init__()
        
        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(1024, 196))
        self.fc2 = nn.Sequential(nn.Linear(196, 64))
        self.fc3 = nn.Sequential(nn.Linear(64, 196))
        self.fc4 = nn.Sequential(nn.Linear(196, 1024))   # Output = 1x32x32 

        # activation function
        if use_activation:  
            self.activation = ReLU()
        else:
            self.activation = nn.Identity()

        self.fc1.append(self.activation)
        self.fc2.append(self.activation)
        self.fc3.append(self.activation)
        self.fc4.append(nn.Sigmoid())
        self.fc4.append(nn.Flatten())           
            
    def forward(self, x): 
        x = self.fc1(x) 
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)

# To load a (untrained) model
def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    return model

# Calculate validation_loss
def calc_loss(model, data_loader, criterion, device):

    model.eval()    # Set model to evaluation mode
    loss = 0        # Initialize loss

    for batch_idx, (clean_images, noisy_images, labels) in enumerate(data_loader):

            # Flattening the images
            clean_images = clean_images.view(clean_images.shape[0], -1)
            noisy_images = noisy_images.view(noisy_images.shape[0], -1)   

            # move to GPU if possible
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
            outputs = model(noisy_images)   # Forward propagation
            loss += criterion(outputs, clean_images).item() # Calculating loss

    return loss / len(data_loader)
            

# Train model function
def train(model, train_loader, valid_loader, optimizer, criterion, epochs, device, write_to_file=False):

    # Keeping the track of loss
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):

        model.train()   # Set model to train mode
        train_loss = 0  # Initialize loss

        for batch_idx, (clean_images, noisy_images, labels) in enumerate(train_loader):

            # Flattening the images
            clean_images = clean_images.view(clean_images.shape[0], -1)
            noisy_images = noisy_images.view(noisy_images.shape[0], -1)                   

            # Move to GPU if possible
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
            optimizer.zero_grad()   # clear the gradients

            # Forward propagation
            outputs = model(noisy_images)               
            loss = criterion(outputs, clean_images)

            # Backprop, updating weights
            loss.backward()
            optimizer.step()

            train_loss += loss.item()   # add loss to the total loss

            # Print training loss
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(clean_images), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        # Calculate training loss
        train_loss /= len(train_loader)

        # Calculate validation loss
        valid_loss = calc_loss(model, valid_loader, criterion, device)

        print("Epoch: {}/{} ".format(epoch+1, epochs),
                "Training Loss: {:.3f} ".format(train_loss),
                "Validation Loss: {:.3f}".format(valid_loss))

        # Append losses for further processings
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    # Write the model parameters to a file
    if write_to_file:
        torch.save(model.state_dict(), "mod_param.pth")

def reshaping(images):
    # Reshaping the images to be 32x32x1
    return torch.reshape(images, (images.shape[0], 1, 32, 32))

if __name__=="__main__":
     
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 100
    
    # Get dataloader from modified MNIST_dataloader
    train_loader, val_loader, test_loader = create_dataloaders(data_dir, batch_size=batch_size)

    # Printing dataset lengths
    print("# of Train set length: ", len(train_loader.dataset))
    print("# of Validation set : ", len(val_loader.dataset))
    print("# of Test set length: ", len(test_loader.dataset))     

    
    model = Autoencoder(use_activation=True)    # define the model
    criterion = nn.MSELoss()                    # define the loss function

    # Use SGD or Adam optimisers
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Defining the device (if there is a GPU or not)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    model.to(device)    # Move model to device (CPU/GPU)

    #===========Predictions on untrained model==========#
    examples = enumerate(test_loader)
    _, (pure_example, noisy_example, labels_example) = next(examples)
    noisy_example = noisy_example.to(device)
    raw_predict = model(noisy_example)  # Predict
    # Move back to CPU   
    raw_predict = raw_predict.detach().cpu()
    noisy_example = noisy_example.detach().cpu()
    # Converting it back to 32x32 images
    raw_predict = reshaping(raw_predict)
    #===========Predictions on untrained model==========#

    # Train the model 
    model, train_losses, valid_losses = train(model, train_loader, val_loader, 
                                              optimiser, criterion, num_epochs, 
                                              device, write_to_file=True)
     
    # ============Prediction on trained model=============#     
    # model = load_model(model, "model_params.pth") # Loading trained model
    processed_predict = model(noisy_example.to(device))
    processed_predict = processed_predict.detach().cpu()
    processed_predict = reshaping(processed_predict) 
    # ============Prediction on trained model=============#
