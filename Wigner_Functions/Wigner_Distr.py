import numpy as np
import matplotlib.pyplot as plt

def rect(x,X,a):
	L = np.where(np.abs(x+X/2)>(a/2),0,1)
	R = np.where(np.abs(x-X/2)>(a/2),0,1)
	return L*R

a = 1
n = 1000
x = np.linspace(-a,a,n)

yffts = np.zeros((len(x),n)).astype(complex)
xfft  = np.fft.fftshift(np.fft.fftfreq(n))
Xs    = np.linspace(-a,a,n)
ys    = np.zeros((len(x),n)).astype(complex)

i=0
for X in Xs:

	y          = rect(x,X,a)
	ys[i,:]    = y
	yffts[i,:] = np.fft.fftshift(np.fft.fft(np.fft.fftshift(y),norm='forward'))

	i+=1

plt.plot(np.abs(yffts[int(n/2),:])**2)
plt.show()

plt.imshow(np.abs(yffts)**2)#,extent=[Xs[0],Xs[-1],xfft[-1],xfft[0]])
#plt.imshow(np.where(np.abs(yffts)==0,1,0),extent=[xfft[0],xfft[-1],Xs[-1],Xs[0]])
plt.colorbar()
plt.show()


xxs, XXs = np.meshgrid(x,Xs)

xs, Xs = np.meshgrid(xfft, Xs)

plt.contour(xxs,XXs,np.abs(ys)**2,[0])
plt.show()

plt.contour(xs,Xs,np.abs(yffts)**2,[0])
plt.show()



