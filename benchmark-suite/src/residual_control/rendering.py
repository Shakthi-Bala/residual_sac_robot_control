import mujoco
import numpy as np


class ArmRenderer:
    def __init__(self, model, width=480, height=360):
        self.model = model
        self.width, self.height = width, height
        self.gl = mujoco.GLContext(width, height)
        self.gl.make_current()
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        self.scene = mujoco.MjvScene(model, maxgeom=2000)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:] = [.35, .05, .4]
        self.camera.distance = 1.15
        self.camera.azimuth = 135
        self.camera.elevation = -15
        self.option = mujoco.MjvOption()
        self.model.vis.headlight.ambient[:] = [.5, .5, .5]
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        self.rgb = np.empty((height, width, 3), dtype=np.uint8)

    def render(self, data):
        self.gl.make_current()
        mujoco.mjv_updateScene(self.model, data, self.option, None, self.camera,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(self.viewport, self.scene, self.context)
        mujoco.mjr_readPixels(self.rgb, None, self.viewport, self.context)
        return self.rgb[::-1].copy()

    def close(self):
        self.context.free()
        self.gl.free()
