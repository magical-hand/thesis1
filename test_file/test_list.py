a=[[1*i for i in range(10)] for _ in range(10)]
print(a)
a[1].extend([11,12])
print(a)
for n in range(len(a)):
    a[n].extend([11,12])
print(a)
for j in a:
    j.append([13,14])
print(a)

data_set=list(range(10,100,5))
index=5
c=[data_set[i] if isinstance(index,list) else [index] for i in index]
print(c)

