import torch
a=[torch.randn([3]) for i  in range(4)]

c=torch.cat(a)
print(c)

for num, i in enumerate(c):
    if i>2:
        print(num,i)