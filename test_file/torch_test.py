import torch
a=[torch.ones([3]) for _  in range(4)]

c=torch.cat(a)
print(c)

c=[sen for sen in a]
print(c)
# # b=[torch.tensor(range(12))]
# # c=a+b
# d=torch.cat(c)
# # print(d)

model=torch.nn.Linear(5,11)
torch.save(model.state_dict(),r'resources\output_result.pt')