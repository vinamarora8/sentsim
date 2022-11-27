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





# pred = torch.sum(op, 1) 

# print(pred)
# pred = pred.to('cpu')
# print(match)
# err = (match - pred).abs()


# # sim = sim.to('cpu')
# # pred = (sim > 0.8).int()
# # err = (match - pred).abs().sum() / N
# # err = err.item()
# # print(err)

# plt.scatter(match, pred)
# plt.show()
