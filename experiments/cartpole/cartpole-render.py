import pickle

import numpy as np
import pyglet
from pyglet import clock
from pyglet.window import key
from rltools.cartpole import Cartpole

from itertools import chain

class TrivialPolicy(object):
    def __init__(self, action):
        self.action = action
    def __call__(self, state):
        return self.action

class DiscreteToContPi(object):
    def __init__(self, actions, pi):
        self.pi=pi
        self.actions = actions
    def __call__(self, state):
        return self.actions[self.pi(state)]

class IdenStateAction(object):
    def __init__(self, proj, actions):
        self.proj = proj
        self.actions = actions
        self.size = proj.size

    def __call__(self, s, a = None):
        if s is None:
            return None
        if a is None:
            return np.vstack([ self.proj(np.hstack((s,act))) for act in self.actions])
        else:
            return self.proj(np.hstack((s,a)))


filename = 'cartpole-test-gus.data'

num_points = 200
angles = 2.0*np.pi*np.arange(num_points)/num_points
x = np.cos(angles)
y = np.sin(angles)
verts = list(chain(*zip(x,y)))
circle = pyglet.graphics.vertex_list(num_points, ('v2f', verts))


cartpole = Cartpole(m_c = 1,
                 m_p = 1,
                 l = 1,
                 g = 9.81,
                 x_damp = 0.1,
                 theta_damp = 0.1,)
cartpole.start_state[:] = [-0.01*np.random.rand(1),0.01,0,0]
cartpole.reset()

cartpole.action_range[0][:] = -3
cartpole.action_range[1][:] = 3
u = np.zeros(1)

time =0

render_circle = False
render_lines = True

# name = 'sarsa2'
# with open('agent-'+name+'.data', 'rb') as f:
#     (phi, policy, agent) = pickle.load(f)
mode = 0

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


label = pyglet.text.Label(str(cartpole.state[0]),
                      font_name='Times New Roman',
                      font_size = 36,
                      x=10, y=10)
                      
sample_traj = []
current_traj = []

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

def draw_cartpole(cartpole):
    theta = np.degrees(cartpole.state[1])
    l = cartpole.l
    
    x = cartpole.state[0] - int(cartpole.state[0])
    
    if render_lines:
        hori_lines = list(chain(*[(i-x, -1, i-x, 1) for i in xrange(-10, 10)]))
        pyglet.graphics.draw(40, pyglet.gl.GL_LINES,
                                 ('v2f', hori_lines),
                                 ('c4B', (255,100,100,255)*40))  
    if render_circle:                         
        circle.draw(pyglet.gl.GL_LINE_LOOP)
    
    pyglet.graphics.draw(4, pyglet.gl.GL_LINE_LOOP,
                             ('v2f', (-0.2,-0.1, 0.2, -0.1, 0.2, 0.1, -0.2, 0.1)),
                             ('c4B', (255,255,255,255)*4))
                             
    pyglet.gl.glPushMatrix()
    pyglet.gl.glRotated(-90, 0,0,1)
    pyglet.gl.glRotated(theta, 0,0,1)
    pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
                             ('v2f', (0,0, l,0)),
                             ('c4B', (255,255,255,255)*2))
    pyglet.graphics.draw(1, pyglet.gl.GL_POINTS,
                             ('v2f', (0,0)),
                             ('c4B', (100,100,255,255)))
    pyglet.gl.glPopMatrix()
    label.text = str(cartpole.state[0])
    
    
    pyglet.gl.glPushMatrix()
    pyglet.gl.glLoadIdentity()

    
    pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
    pyglet.gl.glPushMatrix()
    
    pyglet.gl.glLoadIdentity()
    pyglet.gl.glOrtho(0, window.width, 0, window.height, -1, 1)
    label.draw()
    pyglet.gl.glPopMatrix()
    pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)
    pyglet.gl.glPopMatrix()

@window.event
def on_draw():
    window.clear()
    draw_cartpole(cartpole)

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
    global time
    dt = 1.0/20
    time += dt
    cartpole.dt[-1] =  dt#1.0/100
    old_state = cartpole.state.copy()
    cartpole.step(u)
    new_state = cartpole.state.copy()
    current_traj.append( (old_state, np.clip(u, cartpole.action_range[0], cartpole.action_range[1]), new_state))

def on_key_press(symbol, modifiers):
    global u, mode, time, current_traj
    f = 10
    if symbol == key.RIGHT:
        u += f
    if symbol == key.LEFT:
        u += -f
    if symbol == key.R:
        cartpole.start_state[:] = [-(0.05)*(np.random.rand(1)-0.5),
                             (np.random.rand(1)-0.5)*0.5,
                             (np.random.rand(1)-0.5)*0.5,
                             (np.random.rand(1)-0.5)*0.5]
        sample_traj.append(current_traj)
        current_traj = list()
        cartpole.reset()
        time = 0
    if symbol == key.A:
        mode = 1 if mode != 1 else 0
    if symbol == key.S:
        mode = 2 if mode != 2 else 0
        
    if symbol == key.S:
        with open(filename, 'wb') as f:
            pickle.dump(sample_traj, f)


    print u
def on_key_release(symbol, modifiers):
    global u,render_circle,render_lines
    f=10
    if symbol == key.RIGHT:
        u += -f
    if symbol == key.LEFT:
        u += f
        
    if symbol == key.C:
        render_circle = not render_circle
        
    if symbol == key.L:
        render_lines = not render_lines

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

    clock.schedule_interval(update, 1.0/20.0)
    pyglet.app.run()
    clock.unschedule(update)