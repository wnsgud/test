a = int(input())

b = input()
b = b.split(' ')

c = input()
c = c.split(' ')

d = 0

for i in range(len(b)):

    b[i] = int(b[i])

for i in range(len(b)):
    for j in range(len(b)-i - 1):
        if(b[j] > b[j+1]):
            b[j],b[j+1] = b[j+1],b[j]

for i in range(len(c)):

    c[i] = int(c[i])

for i in range(len(c)):
    for j in range(len(c)-i - 1):
        if(c[j] < c[j + 1]):
            c[j],c[j+1] = c[j+1],c[j]

for x in range(len(c)):
    d = d + b[x] * c[x]
print(d)