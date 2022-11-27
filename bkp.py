# # for epoch in range(20):
# #     print(f'EPOCH {epoch}')


# #     for i, data in enumerate(train_loader):

# #         optimizer.zero_grad()

# #         sent, y = data
# #         outputs = model.forward(sent)
# #         y = y.to(device).float()
# #         loss = criterion(outputs.reshape(-1), y.reshape(-1))
# #         loss.backward()
# #         optimizer.step()

# #         if i % 10 == 0:
# #             print(f'Iteration {i}: ', end='')
# #             print(loss.item())
# #     lr_sched.step()


# # err = 0.0
# # for i, data in enumerate(train_loader):

# #     model.eval()

# #     sent, y = data
# #     y = y.to(device)

# #     outputs = model.forward(sent)
# #     outputs = (outputs > 0.5).float()

# #     err += (y - outputs).abs().mean() / len(train_loader)

# # print(err.item())