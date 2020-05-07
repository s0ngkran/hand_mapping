import matplotlib.pyplot as plt 
print('start program')

print('reading')
with open('a','r') as f:
    dat = f.readlines()

print('loss appending')
loss = []
for i in range(len(dat)):
    if dat[i][:6] == 'epoch=':
        try:
            d = dat[i].split('loss= ')[1][:-1]
            loss.append(int(d))
        except:
            pass

print('ploting')
plt.plot(loss)
plt.show()
plt.plot(loss[-800:])
plt.show()
    