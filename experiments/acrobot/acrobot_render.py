import numpy as np
import pyglet
import pickle
from pyglet import clock
from pyglet.window import key
from rltools.acrobot import Acrobot



acrobot = Acrobot(random_start = False, m1 = 1, m2 = 1, l1 = 1, l2=2, b1=0.1, b2=0.1)
acrobot.start_state[:] = [-0.1*np.random.rand(1),0,0,0]
acrobot.reset()
acrobot.action_range[0][:] = -10
acrobot.action_range[1][:] = 10
u = np.zeros(1)

controller = acrobot.get_swingup_policy()
name = 'cosr2'
with open('agent-'+name+'.data', 'rb') as f:
    (phi, valuefn, policy,agent) = pickle.load(f)
mode = 2

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
    l1 = acrobot.l1
    l2 = acrobot.l2
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
    acrobot.dt[-1] =  dt#1.0/60
    if mode == 1:
        acrobot.step(controller(acrobot.state))
    elif mode == 2:
        acrobot.step(agent.proposeAction(acrobot.state))
    else:
        acrobot.step(u)

def on_key_press(symbol, modifiers):
    global u, mode
    f = 10
    if symbol == key.RIGHT:
        u += f
    if symbol == key.LEFT:
        u += -f
    if symbol == key.R:
        acrobot.start_state[:] = [- 0.2*np.random.rand(1) + 0.1,0,0,0]
        acrobot.reset()
    if symbol == key.A:
        mode = 1 if mode != 1 else 0
    if symbol == key.S:
        mode = 2 if mode != 2 else 0

    print u
def on_key_release(symbol, modifiers):
    global u
    f=10
    if symbol == key.RIGHT:
        u += -f
    if symbol == key.LEFT:
        u += f
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