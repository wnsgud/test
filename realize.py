from torch.utils.data.dataset import Dataset
import torch
from fc_model import NkModel
from class1 import NkDataSet

#Data Load
csv_path = './file/jun.csv'

custom_dataset = NkDataSet(csv_path)

my_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                               batch_size=2,
                                               shuffle=False,
                                               num_workers=1)


print(len(custom_dataset))
#Model Load
#input,hidden,output size
D_in = 30000
#(100 * 100 * 3)
H = 1000
D_out = 7

model = NkModel(D_in, H, D_out)

criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)#1/10000

for t in range(500):

    for i,data in enumerate(my_dataset_loader,0):
        #Forward pass: Compute predicted y by passing x to the model

        images,label = data

        images = images.view(2,30000)
        print(images.size())
        y_pred = model(images)

        print(label)#Compute and print loss
        loss = criterion(y_pred,label)

        print(t,loss.item())

        #Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()