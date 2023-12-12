import torch
a=[torch.randn([3])]

c=torch.cat(a)
print(c)

d=torch.nn.Linear(c.size()[0],2)
for num in d.parameters():
    print(num)

f=d(c)

print(f)