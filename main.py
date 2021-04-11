from model import CycleGAN
from dataloader import DataLoader

#param
fixed_learning_rate_epoches = 100
linearly_decay_learning_learning_rate_epoches = 100
save_model_freq = 5
print_loss_freq = 100
batch_size = 1



dataloader = DataLoader(batch_size)
model = CycleGAN()

total_iter_number = 0

for epoch in range(fixed_learning_rate_epoches+linearly_decay_learning_learning_rate_epoches):
    model.update_learning_rate()
    for i, data in enumerate(dataloader):  # inner loop within one epoch
        total_iter_number += 1
        model.set_input(data)
        model.optimize_parameters()
        if total_iter_number % print_loss_freq == 0:
            loss = model.get_current_losses()
            print("epoch = %d, loss= %f" %(epoch, loss))
    if epoch % save_model_freq == 0:
        model.save_networks()


