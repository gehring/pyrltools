from rltools.acrobot import Acrobot, get_trajectories, get_qs_from_traj,\
    compute_acrobot_from_data

import pyglet
from pyglet import clock
from pyglet.window import key
import numpy as np

def get_U_matrix(q,
                  qdot, 
                  qdotdot, 
                  y):
    c = np.cos(q)
    c12 = np.sin(np.sum(q, axis=1))
     
    c1 = np.sin(q[:,0])
    c2 = c[:,1]
    s2 = np.sin(q[:,1])
    
    
    qd1 = qdot[:,0]
    qd2 = qdot[:,1]
    qdd1 = qdotdot[:,0]
    qdd2 = qdotdot[:,1]
    
    u = np.empty((q.shape[0], 7))
    u[:,0] = qdd1
    u[:,1] = 3*c2*qdd1 + s2*qd1**2 + c2*qdd2 - s2*qd2**2 - 2*s2*qd2*qd1
    u[:,2] = 2*qdd2 + qdd1
    u[:,3] = c1
    u[:,4] = c12*2
    u[:,5:] = qdot
    return u

domain = Acrobot(random_start = False, 
                 m1 = 1, 
                 m2 = 1, 
                 l1 = 1, 
                 l2=2, 
                 b1=0.1, 
                 b2=0.1)
domain.start_state[0] = 0.01
domain.dt[-1] = 0.01
domain.action_range = [np.array([-10]), np.array([10])]

print 'generating trajectories...'
c = domain.get_swingup_policy()
c.energyshaping.k1 = 2.0
c.energyshaping.k2 = 1.0
c.energyshaping.k3 = 0.1
alph =0.0
controller = lambda q: alph*c(q) + (1-alph)*np.random.rand()*20-10
states, torques = get_trajectories(domain, 1, 10000, controller = controller)
q, qd, qdd, y = get_qs_from_traj(states, torques, domain.dt[-1])
qdd = np.vstack((domain.state_dot(np.hstack((q[i,:], qd[i,:])), 0, y[i])[2:] for i in xrange(q.shape[0])))
id_domain = compute_acrobot_from_data(q, qd, qdd, y, random_start = False)
# q[:,0] = np.remainder(q[:,0] - np.pi/2, 2*np.pi)
U = get_U_matrix(q, qd, qdd, y)
print 'solving system id...'
# print q[:10,:]
# print qd[:10,:]
# print qdd[:10,:]
# 
# print U[:10, :]
# print y[:10]

m1 = domain.m1
m2 = domain.m2
I1 = domain.Ic1
I2 = domain.Ic2
l1 = domain.l1
l2 = domain.l2
lc1 = domain.lc1
lc2 = domain.lc2
g = domain.g
b1, b2 = domain.b1, domain.b2

a = np.array([m1*lc1**2 + m2*l1**2 + m2*lc2**2+ I1 + I2,
              m2*l1*lc2,
              m2*lc2**2 + I2,
              (m1*lc1 + m2*l1)*g,
              m2*lc2*g,
              b1,
              b2])


# print qdd[0,:]
# print domain.state_dot(np.hstack((q[0,:], qd[0,:])), 0, y[0])
# print y[:2]
 
# print U[:2,:].dot(a)
# print np.linalg.norm(U.dot(a) - y)
# print np.linalg.norm(U.dot(id_domain.a) - y)
# print 'Cond and Det:'
# print np.linalg.cond(U)
# print np.linalg.det(U.T.dot(U))
# print 'Results:'
# print np.allclose(a, id_domain.a)
print a
print id_domain.a
# i = 0
# print qd[i,:], qdd[i,:]
# print id_domain.state_dot(np.hstack((q[i,:], qd[i,:])), 0, y[i])
# sys.exit()
mode = 1
u=0
domain.random_start = False
domain.reset()
acrobot = domain #id_domain
acrobot.start_state[0] = 0.005

controller = id_domain.get_swingup_policy()
# controller = domain.get_swingup_policy()

configTemp = pyglet.gl.Config(sample_buffers=1,
    samples=4,
    double_buffer=True,
    alpha_size=0)

platform = pyglet.window.get_platform()
display = platform.get_default_display()
screen = display.get_default_screen()

try:
    config= screen.get_best_config(configTemp)
except:
    config=pyglet.gl.Config(double_buffer=True)

window = pyglet.window.Window(config=config, resizable=True)


def get_mouse_coord(x, y):
        vp = (pyglet.gl.GLint * 4)()
        mvm = (pyglet.gl.GLdouble * 16)()
        pm = (pyglet.gl.GLdouble * 16)()

        pyglet.gl.glGetIntegerv(pyglet.gl.GL_VIEWPORT, vp)
        pyglet.gl.glGetDoublev(pyglet.gl.GL_MODELVIEW_MATRIX, mvm)
        pyglet.gl.glGetDoublev(pyglet.gl.GL_PROJECTION_MATRIX, pm)

        wx = pyglet.gl.GLdouble()
        wy = pyglet.gl.GLdouble()
        wz = pyglet.gl.GLdouble()

        pyglet.gl.gluUnProject(x, y, 0, mvm, pm, vp, wx, wy, wz)
        mcoord = (wx.value, wy.value)

        return mcoord

def draw_acrobot(acrobot):
    theta = np.degrees(acrobot.state[:2])
#     l1 = acrobot.l1
#     l2 = acrobot.l2
    l1 = 1
    l2 = 2
    pyglet.gl.glPushMatrix()
    pyglet.gl.glRotated(-90, 0,0,1)
    pyglet.gl.glRotated(theta[0], 0,0,1)
    pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
                             ('v2f', (0,0, l1,0)),
                             ('c4B', (255,255,255,255)*2))
    pyglet.graphics.draw(1, pyglet.gl.GL_POINTS,
                             ('v2f', (0,0)),
                             ('c4B', (100,100,255,255)))
    pyglet.gl.glTranslated(l1,0,0)
    pyglet.gl.glRotated(theta[1], 0,0,1)
    pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
                             ('v2f', (0,0, l2,0)),
                             ('c4B', (255,255,255,255)*2))
    pyglet.graphics.draw(1, pyglet.gl.GL_POINTS,
                             ('v2f', (0,0)),
                             ('c4B', (100,100,255,255)))
    pyglet.gl.glPopMatrix()


@window.event
def on_draw():
    window.clear()
    draw_acrobot(acrobot)

@window.event
def on_resize(width, height):
    pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
    pyglet.gl.glLoadIdentity()
    pyglet.gl.glViewport(0, 0, width, height)
    rangex = (-4,4)
    rangey = (-4,4)
    ratio = float(height)/width
    lx = rangex[1] - rangex[0]
    ly = rangey[1] - rangey[0]

    if lx*ratio >= ly:
        dy = lx*ratio - ly
        pyglet.gl.glOrtho(rangex[0], rangex[1], rangey[0]- dy/2, rangey[1]+dy/2, -1, 1)
    else:
        dx = ly/ratio - lx
        pyglet.gl.glOrtho(rangex[0]-dx/2, rangex[1] + dx/2, rangey[0], rangey[1], -1, 1)
    pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)
    return True

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    (mx, my)= get_mouse_coord(x, y)
    pyglet.gl.glTranslatef(mx, my, 0)
    pyglet.gl.glScalef(1.05**scroll_y, 1.05**scroll_y, 1)
    pyglet.gl.glTranslatef(-mx, -my, 0)

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    mcoord1 = get_mouse_coord(x, y)
    mcoord2 = get_mouse_coord(x + dx, y+ dy)
    pyglet.gl.glTranslatef(mcoord2[0] - mcoord1[0], mcoord2[1] - mcoord1[1], 0)

def update(dt):
    acrobot.dt[-1] =  dt#1.0/100
    if mode == 1:
        acrobot.step(controller(acrobot.state))
    else:
        acrobot.step(u)

def on_key_press(symbol, modifiers):
    global u
    f = 10
    if symbol == key.RIGHT:
        u += f
    if symbol == key.LEFT:
        u += -f

    print u
def on_key_release(symbol, modifiers):
    global u, mode
    f=10
    if symbol == key.RIGHT:
        u += -f
    if symbol == key.LEFT:
        u += f
        
    if symbol == key.A:
        mode = 1 if mode != 1 else 0
        print 'mode '+str(mode)
        
    if symbol == key.R:
        acrobot.reset()
        
    print u

window.push_handlers(on_key_press)
window.push_handlers(on_key_release)



if __name__ == '__main__':
    pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
    pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
    pyglet.gl.glEnable(pyglet.gl.GL_LINE_SMOOTH )
    pyglet.gl.glEnable(pyglet.gl.GL_POLYGON_SMOOTH )
    pyglet.gl.glEnable(pyglet.gl.GL_POINT_SMOOTH )
    pyglet.gl.glClearColor(0, 0, 0, 1.0)
    pyglet.gl.glLineWidth(3)
    pyglet.gl.glPointSize(6)

    height = window.height
    width = window.width

    pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
    pyglet.gl.glLoadIdentity()
    pyglet.gl.glViewport(0, 0, width, height)
    rangex = (-4,4)
    rangey = (-4,4)
    ratio = float(height)/width
    lx = rangex[1] - rangex[0]
    ly = rangey[1] - rangey[0]

    if lx*ratio >= ly:
        dy = lx*ratio - ly
        pyglet.gl.glOrtho(rangex[0], rangex[1], rangey[0]- dy/2, rangey[1]+dy/2, -1, 1)
    else:
        dx = ly/ratio - lx
        pyglet.gl.glOrtho(rangex[0]-dx/2, rangex[1] + dx/2, rangey[0], rangey[1], -1, 1)
    pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)

    clock.schedule_interval(update, 1.0/100.0)
    pyglet.app.run()