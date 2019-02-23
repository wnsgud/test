from torch.utils.data.dataset import Dataset
import torch
from cnn_model import Cnn_Model
from my_dataset import NkDataSet


def train(my_dataset_loader,model,criterion,optimizer,epoch):

    model.train()

    for i, data in enumerate(my_dataset_loader, 0):

        images, label = data

        y_pred = model(images)

        loss = criterion(y_pred, label)

        print(epoch, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(my_dataset_loader, model, criterion, epoch):

    model.eval()

    for i, data in enumerate(my_dataset_loader, 0):
        images, label = data

        y_pred = model(images)

        loss = criterion(y_pred, label)

        print(epoch, loss.item())

csv_path = './file/jun.csv'

custom_dataset = NkDataSet(csv_path)

my_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                batch_size=2,
                                                shuffle=False,num_workers=1)


print('end')
csv_path = './file/junn.csv'


custom_dataset = NkDataSet(csv_path)

test_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                  batch_size=2,
                                                  shuffle=False,
                                                  num_workers=1)


D_in = 30000
H = 100
D_out = 7

fc_model = Cnn_Model()

criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(fc_model.parameters(), lr=1e-4)

for epoch in range(500):

    print('epoch',epoch)

    train(my_dataset_loader,fc_model,criterion,optimizer,epoch)
    test(my_dataset_loader,fc_model,criterion,epoch)