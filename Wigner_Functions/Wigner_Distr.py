import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import signal

def rect(x,a):
	return np.where(np.abs(x)>(a/2),0,1)

def triangle(x,a):
	return np.where(np.abs(x)>(a/2),0,-np.abs(x)+1)

def squared_triangle(x,a):
	return np.where(np.abs(x)>(a/2),0,-np.abs(x**2)+1)
	
def sin2(x,a):
	return np.sin(a*x)**2

def square_wave(x,a):
	return signal.square(x)

a = 1
n = 10
extx = 5*a
extX = 5*a


x     = np.linspace(-extx,extx,int(extx/a*n))
Xs    = np.linspace(-extX,extX,int(extX/a*n))
xfft  = np.fft.fftshift(np.fft.fftfreq(len(x)))
ys    = np.zeros((len(Xs),len(x))).astype(complex)
yffts = np.zeros((len(Xs),len(x))).astype(complex)



i=0
for X in Xs:

	y          = square_wave(x-X/2,a)*square_wave(x+X/2,a)#np.sin(x-X/2)*np.sin(x+X/2)
	ys[i,:]    = y
	yffts[i,:] = np.fft.fftshift(np.fft.fft(np.fft.fftshift(y),norm='forward'))

	i+=1

xxs, XXs = np.meshgrid(x,Xs)

plt.imshow(np.real(ys),extent=[x[0],x[-1],Xs[-1],Xs[0]],cmap='BrBG')
plt.xlabel("x'")
plt.ylabel("X")
plt.title("G(x',X)")
plt.colorbar()
plt.show()

# plt.contour(xxs,XXs,np.real(ys))
# plt.xlabel("x'")
# plt.ylabel("X")
# plt.title("G(x',X)")
# plt.show()

# plt.plot(np.abs(yffts[int(n/2),:])**2)
# plt.show()

plt.imshow((np.abs(yffts)**2),extent=[xfft[0],xfft[-1],Xs[-1],Xs[0]],aspect=xfft[0]/Xs[0])#norm=colors.LogNorm()
plt.colorbar()
plt.xlabel("frequency along x (u)")
plt.ylabel("X")
plt.title("Wigner Function Magnitude (log scale)")
plt.show()

plt.imshow((np.angle(yffts)),extent=[xfft[0],xfft[-1],Xs[-1],Xs[0]],aspect=xfft[0]/Xs[0])
plt.colorbar()
plt.xlabel("frequency along x (u)")
plt.ylabel("X")
plt.title("Wigner Function Phase")
plt.show()

# plt.imshow((np.abs(yffts)**2),extent=[xfft[0],xfft[-1],Xs[-1],Xs[0]])
# plt.colorbar()
# plt.xlabel("frequency along x (u)")
# plt.ylabel("X")
# plt.show()



# xs, Xs = np.meshgrid(xfft, Xs)
# plt.contour(xs,Xs,np.abs(yffts)**2,[0])
# plt.show()



