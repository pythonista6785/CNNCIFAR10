import sys
import Utils
from CNNModel import CNNModel
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


def main():
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(device)
    path_data = './data/' # for storing the dataset
    batch_size = 16
    num_epochs = 100
    trainds, trainloader, testds, testloader = Utils.prepare_data(path_data, batch_size)
    train_iter = iter(trainloader)
    images, labels = next(trainloader)    # get a batch of data e.g., 16X3X32X32
    print(images[0].shape)    
    d_class2idx = trainds.class_to_idx     # dictionary of class names and their ids
    print(d_class2idx)
    d_idx2class = dict(zip(d_class2idx.values(), d_class2idx.keys()))
    print(d_idx2class)
    images, labels = next(train_iter)     # get a batch of training data
    Utils.plot_image(images, labels)      # plot images 

    # print image class names 
    print(' '.join('%5' % d_idx2class[int(labels[j])] for j in range(len(images))))
    plt.show()

    net = CNNModel()   # create the model 
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    trainloader = torch.utils.data.DataLoader(trainds, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    running_loss = 0
    printfreq = 1000
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs) # forward pass 
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % printfreq == printfreq-1:
                print('epoch:',epoch, i+1, running_loss/printfreq)
                running_loss = 0
        for param_tensor in net.state_dict():
            print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        print(optimizer.state_dict().keys())
        print(optimizer.state_dict()['param_groups'])

        fname = './models/CIFAR10_cnn.pth'
        torch.save(net.state_dict(), fname)   # save model with state dictionary

        # -- load saved model from state dictionary 
        loaded_dict = torch.load(fname)
        net.load_state_dict(loaded_dict)

        net.eval()
        fname = './models/CIFAR10_cnn.pth'
        loaded_dict = torch.load(fname)
        net.load_state_dict(loaded_dict)
        print('actual test images..')
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        Utils.plot_image(images, labels)
        print(' '.join('%5s' % d_idx2class[int(labels[j])] for j in range(len(images))))
        plt.show()

        print("-------predictions--------")
        preds = net(images)
        preds = outputs.argmax(dim=1)
        Utils.plot_images(images, preds)
        print(' '.join('%5s' % d_idx2class[int(preds[j])] for j in range(len(images))))
        print(plt.show())

        #--------Compute accuracy on trained model-------
        total = 0   # keeps track of how many images we have processed
        correct = 0  # keeps track of how many correct images or net predicts 
        with torch.no_grad():
            for i, data in enumerate(testloader):
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size()[0]
                correct += (predicted == labels).sum().item()
        print("Accuracy: ", correct/total)

if __name__ == "__main__":
    sys.exit(int(main() or 0))