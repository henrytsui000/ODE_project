import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from ipywidgets import IntProgress
from IPython.display import display
from matplotlib.animation import FuncAnimation

L1, L2 = 1, 1
m1, m2 = 1, 1
g = 9.81

def deriv(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)

    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
             
    return theta1dot, z1dot, theta2dot, z2dot

def calc_E(y):
    th1, th1d, th2, th2d = y.T
    V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
    T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
            2*L1*L2*th1d*th2d*np.cos(th1-th2))
    return T + V

tmax, dt = 100, 0.01
t = np.arange(0, tmax+dt, dt)

y0 = np.array([0*np.pi/7, 0, 3*np.pi/12, 0])
y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))


EDRIFT = 0.05
E = calc_E(y0)
if np.max(np.sum(np.abs(calc_E(y) - E))) > EDRIFT:
    sys.exit('Maximum energy drift of {} exceeded.'.format(EDRIFT))

theta1, theta2 = y[:,0], y[:,2]
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

r = 0.05
# Plot a trail of the m2 bob's position for the last trail_secs seconds.
trail_secs = 1
max_trail = int(trail_secs / dt)

upper_pos = []
lower_pos = []
upper_the = []
lower_the = []

def make_plot(i):
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
    c0 = Circle((0, 0), r/2, fc='k', zorder=10)
    c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10)
    c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)
    c3 = Circle((x1[i], -2), r, fc='b', ec='b', zorder=10)
    c4 = Circle((x2[i], -2), r, fc='r', ec='r', zorder=10)
    # print(f"{x1[i]}, {y2[i]}")

    upper_pos.append(x1[i])
    lower_pos.append(x2[i])
    upper_the.append(np.arctan(x1[i]/y1[i]))
    lower_the.append(np.arctan(x2[i]/y2[i]))

    
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.add_patch(c3)
    ax.add_patch(c4)

    # The trail will be divided into ns segments and plotted as a fading line.
    ns = 20
    s = max_trail // ns

    for j in range(ns):
        imin = i - (ns-j)*s
        if imin < 0:
            continue
        imax = imin + s + 1
        # The fading looks better if we square the fractional length along the trail.
        alpha = (j/ns)**2
        ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
                lw=2, alpha=alpha)

    # Centre the image on the fixed anchor point, and ensure the axes are equal
    ax.set_xlim(-L1-L2-r, L1+L2+r)
    ax.set_ylim(-L1-L2-r, L1+L2+r)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('frames/img{:03d}.png'.format(i//di), dpi=72)
    plt.cla()

fps = 10
di = int(1/fps/dt)
fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
ax = fig.add_subplot(111)

for i in range(0, t.size, di):
    make_plot(i)

fig, ax = plt.subplots()
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)
line, = ax.plot(0, 0)

up_data = []
low_data = []


def animation_frame(idx):
    up_data.append(upper_the[idx])
    low_data.append(lower_the[idx])
    line.set_xdata(up_data)
    line.set_ydata(low_data)
    return line,

animation = FuncAnimation(fig, func=animation_frame, frames=range(300), interval=10)
plt.show()

