import pyglet
import numpy as np
from pyglet import clock
from rltools.acrobot import Acrobot


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

acrobot = Acrobot()


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
    theta = acrobot.state[:2]
    l1 = acrobot.l1
    l2 = acrobot.l2
    pyglet.gl.glPushMatrix()
    pyglet.gl.glRotated(theta[0], 0,0,1)
    pyglet.graphics.draw(1, pyglet.gl.GL_LINES,
                             ('v2f', ((0,0), (l1,0))),
                             ('c4B', ((255,255,255,255))))
    pyglet.graphics.draw(1, pyglet.gl.GL_POINTS,
                             ('v2f', ((0,0))),
                             ('c4B', ((100,100,255,255))))
    pyglet.gl.glTranslated(l1,0,0)
    pyglet.gl.glRotated(theta[1], 0,0,1)
    pyglet.graphics.draw(1, pyglet.gl.GL_LINES,
                             ('v2f', ((0,0), (l2,0))),
                             ('c4B', ((255,255,255,255))))
    pyglet.graphics.draw(1, pyglet.gl.GL_POINTS,
                             ('v2f', ((0,0))),
                             ('c4B', ((100,100,255,255))))
    pyglet.gl.glPopMatrix()


@window.event
def on_draw():
    window.clear()

@window.event
def on_resize(width, height):
    pyglet.gl.glViewport(0, 0, width, height)
    pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
    pyglet.gl.glLoadIdentity()
    pyglet.gl.glOrtho(0, 10, 0, 10, -1, 1)
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
    acrobot.dt[0] = dt
    acrobot.step(np.zeros(1))

if __name__ == '__main__':
    pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
    pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
    pyglet.gl.glEnable(pyglet.gl.GL_LINE_SMOOTH )
    pyglet.gl.glEnable(pyglet.gl.GL_POLYGON_SMOOTH )
    pyglet.gl.glEnable(pyglet.gl.GL_POINT_SMOOTH )
    pyglet.gl.glClearColor(0, 0, 0, 1.0)
    pyglet.gl.glLineWidth(3)
    pyglet.gl.glPointSize(6)
    clock.schedule_interval(update, 1/10.0)
    pyglet.app.run()