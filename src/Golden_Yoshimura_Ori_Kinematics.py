"""
Golden Ratio Yoshimura Kinematics

This code implements the forward kinematics of the Golden Ratio Yoshimura structure and provides visualization tools.

Author: Yogesh Phalak

License:
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import vpython as vp
from scipy.spatial.transform import Rotation as R


class BaseModule:
    """
    Base class for a module of the Golden Ratio Yoshimura structure.
    """

    def __init__(self, base=None, scene=None, phase=0):
        """
        Initializes the BaseModule.

        Parameters:
        - base (numpy.ndarray, optional): Base vertex positions. Default is None.
        - scene (vpython.canvas, optional): VPython scene for visualization. Default is None.
        - phase (int, optional): Phase of the module. Default is 0.
        """

        # The Golden Ratio
        self.varphi = (1 + 5 ** 0.5) / 2

        # Kinematic parameters
        self.theta = 0.0
        self.beta = np.arctan(1 / self.varphi)
        self.l = 1
        self.w = self.l / 2 * np.tan(self.beta)
        self.visualize = scene is not None
        self.scene = scene
        self.x = None
        self.base = base
        self.top = None
        self.top_c = None
        self.base_c = None
        self.T_mat = None

        if self.visualize:
            # Visualization parameters
            self.scene = scene
            self.vp_x = None
            self.edges = None
            self.faces = None
            self.vp_top_c = None
            self.vp_base_c = None
            self.shape = None
            self.colors = VtColorPalette()

        self.phase = int(phase)
        self.n = 12
        self.construct_unit()

    def __del__(self):
        """
        Destructor for the BaseModule.
        :return: None
        """
        if self.scene is not None:
            self.hide_unit()
            self.hide_shape()
            if self.vp_x is not None:
                for i in range(self.n):
                    self.vp_x[i].delete()
                for i in range(4):
                    self.edges[i].delete()
                for i in range(18):
                    self.faces[i].delete()
            if self.shape is not None:
                for i in range(4):
                    self.shape[i].delete()

    def construct_unit(self):
        """
        Constructs the unit module.
        :return: None
        """
        self.geometry()
        self.apply_phase()
        if self.base is not None:
            self.x = forward_kinematics(self.base, self.x[:, 0:3], self.x)[0]
        else:
            self.base = self.x[:, 0:3]
        self.top = self.x[:, self.n - 3:self.n]
        self.get_transformation_matrix()

        self.top_c = self.top.mean(axis=1)
        self.base_c = self.base.mean(axis=1)

        if self.scene is not None and self.visualize:
            self.visualize_unit()

    def geometry(self):
        """
        Calculates the geometry of the module.
        :return:
        """
        pass

    def get_transformation_matrix(self):
        """
        Calculates the transformation matrix.
        :return: Transformation matrix (numpy.ndarray)
        """
        self.T_mat = forward_kinematics(self.top, self.base, np.eye(3))[1]
        return self.T_mat

    def apply_phase(self):
        """
        Applies the phase (gamma) to the module.
        :return:
        """
        self.phase = self.phase % 3
        self.x[:, 0:3] = np.roll(self.x[:, 0:3], self.phase, axis=1)
        self.x[:, 3:9] = np.roll(self.x[:, 3:9], self.phase * 2, axis=1)
        self.x[:, self.n - 3:self.n] = np.roll(self.x[:, self.n - 3:self.n], self.phase, axis=1)

    def get_vp_vertices(self):
        """
        Gets the VPython vertices.
        :return:
        """
        if self.scene is not None:
            self.vp_x = [
                vp.vertex(pos=vp.vector(self.x[0, i], self.x[1, i], self.x[2, i]), color=self.colors.land_grant_grey)
                for i in range(self.n)]
            self.vp_top_c = vp.vertex(pos=vp.vector(self.top_c[0], self.top_c[1], self.top_c[2]), color=vp.color.black)
            self.vp_base_c = vp.vertex(pos=vp.vector(self.base_c[0], self.base_c[1], self.base_c[2]),
                                       color=vp.color.black)

    def show_unit(self):
        """
        Shows the unit module.
        :return:
        """
        if self.scene is not None:
            for i in range(self.n):
                self.vp_x[i].visible = True
            for i in range(4):
                self.edges[i].visible = True
            for i in range(18):
                self.faces[i].visible = True

    def show_shape(self):
        """
        Shows the shape of the module.
        :return:
        """
        if self.scene is not None:
            for i in range(4):
                self.shape[i].visible = True

    def hide_unit(self):
        """
        Hides the unit module.
        :return:
        """
        if self.scene is not None:
            for i in range(self.n):
                self.vp_x[i].visible = False
            for i in range(4):
                self.edges[i].visible = False
            for i in range(18):
                self.faces[i].visible = False

    def hide_shape(self):
        """
        Hides the shape of the module.
        :return:
        """
        if self.scene is not None:
            for i in range(4):
                self.shape[i].visible = False

    def visualize_unit(self):
        """
        Visualizes the unit module.
        :return:
        """
        self.get_vp_vertices()
        crease_radius = 0.0025
        if self.scene is not None:
            self.edges = [
                vp.curve(pos=[self.vp_x[i].pos for i in [0, 1, 2, 0]], color=self.colors.burnt_orange,
                         radius=crease_radius),
                vp.curve(pos=[self.vp_x[i].pos for i in [9, 10, 11, 9]], color=self.colors.burnt_orange,
                         radius=crease_radius),
                vp.curve(pos=[self.vp_x[i].pos for i in [3, 4, 5, 6, 7, 8, 3]], color=self.colors.chicago_maroon,
                         radius=crease_radius),
                vp.curve(pos=[self.vp_x[i].pos for i in [0, 4, 10, 6, 2, 8, 9, 4, 1, 6, 11, 8, 0]],
                         color=self.colors.chicago_maroon, radius=crease_radius)]

            self.faces = [vp.triangle(vs=[self.vp_x[i] for i in [0, 4, 1]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [9, 4, 10]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [1, 6, 2]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [10, 6, 11]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [2, 8, 0]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [11, 8, 9]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [3, 0, 8]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [3, 9, 8]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [3, 0, 4]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [3, 9, 4]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [5, 1, 4]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [5, 10, 4]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [5, 1, 6]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [5, 10, 6]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [7, 2, 6]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [7, 11, 6]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [7, 2, 8]]),
                          vp.triangle(vs=[self.vp_x[i] for i in [7, 11, 8]])]

            x_vec = ((self.vp_x[1].pos + 2 * self.vp_x[0].pos) / 3 - self.vp_base_c.pos).norm()
            y_vec = (self.vp_x[1].pos - self.vp_base_c.pos).norm()
            z_vec = x_vec.cross(y_vec)

            ax_len = 0.25
            x_vec = x_vec * ax_len
            y_vec = y_vec * ax_len
            z_vec = z_vec * ax_len

            self.shape = [vp.curve(pos=[self.vp_top_c.pos, self.vp_base_c.pos], color=vp.color.black, radius=0.01),
                          vp.arrow(pos=self.vp_base_c.pos, axis=x_vec, color=vp.color.red, shaftwidth=0.01),
                          vp.arrow(pos=self.vp_base_c.pos, axis=y_vec, color=vp.color.green, shaftwidth=0.01),
                          vp.arrow(pos=self.vp_base_c.pos, axis=z_vec, color=vp.color.blue, shaftwidth=0.01)]


class VtColorPalette:
    """
    Class defining a color palette for visualizations.
    """

    def __init__(self):
        """
        Initializes the color palette with predefined color vectors.
        """

        # connecting crease
        self.chicago_maroon = vp.vector(134, 31, 65) / 255
        # internal crease
        self.burnt_orange = vp.vector(229, 117, 31) / 255

        self.hokie_stone = vp.vector(117, 120, 123) / 255
        self.yardline_white = vp.vector(255, 255, 255) / 255
        self.pylon_purple = vp.vector(100, 38, 103) / 255
        self.boundless_pink = vp.vector(206, 0, 88) / 255
        self.triumphant_yellow = vp.vector(247, 234, 72) / 255
        self.sustainable_teal = vp.vector(80, 133, 144) / 255
        self.vibrant_turquoise = vp.vector(44, 213, 196) / 255
        # face
        self.land_grant_grey = vp.vector(215, 210, 203) / 255

        self.skipper_smoke = vp.vector(229, 225, 230) / 255
        self.impact_orange = vp.vector(202, 79, 0) / 255
        self.black = vp.vector(0.0, 0.0, 0.0) / 255


class ZeroPopOut(BaseModule):
    """
    Class representing a module with zero pop-outs.
    """

    def __init__(self, phase=0, base=None, scene=None):
        """
       Initializes the ZeroPopOut module.

       Parameters:
       - phase (int, optional): Phase of the module. Default is 0.
       - base (numpy.ndarray, optional): Base vertex positions. Default is None.
       - scene (vpython.canvas, optional): VPython scene for visualization. Default is None.
       """
        self.theta = None
        super().__init__(base, scene, phase)

    def geometry(self):
        """
        Defines the geometry of the ZeroPopOut module.
        """

        self.x = np.zeros((3, 12))
        self.n = self.x.shape[1]
        self.theta = np.arccos(1 / 3 ** 0.5 / np.tan(self.beta))
        r = self.l / 3 ** 0.5

        self.x[:, 0] = np.array([r * np.cos(np.pi / 6), r * np.sin(np.pi / 6), -self.w * np.sin(self.theta)])
        self.x[:, 1] = np.array([-r * np.cos(np.pi / 6), r * np.sin(np.pi / 6), -self.w * np.sin(self.theta)])
        self.x[:, 2] = np.array([0, -r, -self.w * np.sin(self.theta)])

        self.x[:, 4] = np.array([0, r, 0])
        self.x[:, 6] = np.array([-r * np.cos(np.pi / 6), -r * np.sin(np.pi / 6), 0])
        self.x[:, 8] = np.array([r * np.cos(np.pi / 6), -r * np.sin(np.pi / 6), 0])
        self.x[:, 3] = (self.x[:, 4] + self.x[:, 8]) / 2
        self.x[:, 5] = (self.x[:, 4] + self.x[:, 6]) / 2
        self.x[:, 7] = (self.x[:, 6] + self.x[:, 8]) / 2

        self.x[:, 9] = np.array([r * np.cos(np.pi / 6), r * np.sin(np.pi / 6), self.w * np.sin(self.theta)])
        self.x[:, 10] = np.array([-r * np.cos(np.pi / 6), r * np.sin(np.pi / 6), self.w * np.sin(self.theta)])
        self.x[:, 11] = np.array([0, -r, self.w * np.sin(self.theta)])


class OnePopOut(BaseModule):
    """
    Class representing a module with one pop-out.
    """

    def __init__(self, phase=0, base=None, scene=None):
        """
       Initializes the OnePopOut module.

       Parameters:
       - phase (int, optional): Phase of the module. Default is 0.
       - base (numpy.ndarray, optional): Base vertex positions. Default is None.
       - scene (vpython.canvas, optional): VPython scene for visualization. Default is None.
       """
        self.theta = None
        self.alpha = None
        self.phi = None
        super().__init__(base, scene, phase)

    def geometry(self):
        """
        Defines the geometry of the OnePopOut module.
        """
        self.alpha = np.arcsin(1 / np.sqrt(1 + self.varphi ** -2)) - self.beta

        self.x = np.zeros((3, 12))
        self.n = self.x.shape[1]

        self.phi = np.arcsin(2 * np.sin(self.alpha))

        self.x[:, 0] = np.array([1 / 2, self.w * np.cos(self.theta), -self.w * np.sin(self.theta)])
        self.x[:, 1] = np.array([0, np.cos(self.alpha) + 1 / 2 * np.cos(self.phi), -self.w])
        self.x[:, 2] = np.array([-1 / 2, self.w * np.cos(self.theta), -self.w * np.sin(self.theta)])

        self.x[:, 4] = np.array([np.sin(self.alpha), np.cos(self.alpha), 0])
        self.x[:, 5] = np.array([0, np.cos(self.alpha) + 1 / 2 * np.cos(self.phi), 0])
        self.x[:, 6] = np.array([-np.sin(self.alpha), np.cos(self.alpha), 0])
        self.x[:, 8] = np.array([0, 0, 0])
        self.x[:, 7] = (self.x[:, 6] + self.x[:, 8]) / 2
        self.x[:, 3] = (self.x[:, 4] + self.x[:, 8]) / 2

        self.x[:, 9] = np.array([1 / 2, self.w * np.cos(self.theta), self.w * np.sin(self.theta)])
        self.x[:, 10] = np.array([0, np.cos(self.alpha) + 1 / 2 * np.cos(self.phi), self.w])
        self.x[:, 11] = np.array([-1 / 2, self.w * np.cos(self.theta), self.w * np.sin(self.theta)])


class TwoPopOut(BaseModule):
    """
    Class representing a module with two pop-outs.
    """

    def __init__(self, phase=0, base=None, scene=None):
        """
        Initializes the TwoPopOut module.

        Parameters:
        - phase (int, optional): Phase of the module. Default is 0.
        - base (numpy.ndarray, optional): Base vertex positions. Default is None.
        - scene (vpython.canvas, optional): VPython scene for visualization. Default is None.
        """
        self.theta = None
        super().__init__(base, scene, phase)

    def geometry(self):
        """
        Defines the geometry of the TwoPopOut module.
        """
        self.theta = np.arccos((1 - 4 * self.w ** 2) / (2 * self.w * (1 + 4 * self.w ** 2) ** 0.5)) - self.beta
        self.x = np.zeros((3, 12))

        self.x[:, 0] = np.array([0, -self.w * np.cos(self.theta), -self.w * np.sin(self.theta)])
        self.x[:, 1] = np.array([1 / 2, 1 / 2, -self.w])
        self.x[:, 2] = np.array([-1 / 2, 1 / 2, -self.w])

        self.x[:, 3] = np.array([0, 0, 0])
        self.x[:, 4] = np.array([1 / 2, 0, 0])
        self.x[:, 5] = np.array([1 / 2, 1 / 2, 0])
        self.x[:, 6] = np.array([0, 1 / 2, 0])
        self.x[:, 7] = np.array([-1 / 2, 1 / 2, 0])
        self.x[:, 8] = np.array([-1 / 2, 0, 0])

        self.x[:, 9] = np.array([0, -self.w * np.cos(self.theta), self.w * np.sin(self.theta)])
        self.x[:, 10] = np.array([1 / 2, 1 / 2, self.w])
        self.x[:, 11] = np.array([-1 / 2, 1 / 2, self.w])


class ThreePopOut(BaseModule):
    """
    Class representing a module with three pop-outs.
    """

    def __init__(self, phase=0, base=None, scene=None):
        """
       Initializes the ThreePopOut module.

       Parameters:
       - phase (int, optional): Phase of the module. Default is 0.
       - base (numpy.ndarray, optional): Base vertex positions. Default is None.
       - scene (vpython.canvas, optional): VPython scene for visualization. Default is None.
       """
        super().__init__(base, scene, phase)

    def geometry(self):
        """
        Defines the geometry of the ThreePopOut module.
        """
        self.x = np.zeros((3, 12))
        self.n = self.x.shape[1]
        r = self.l / 3 ** 0.5

        self.x[:, 0] = np.array([r * np.cos(np.pi / 6), -r * np.sin(np.pi / 6), -self.w])
        self.x[:, 1] = np.array([0, r, -self.w])
        self.x[:, 2] = np.array([-r * np.cos(np.pi / 6), -r * np.sin(np.pi / 6), -self.w])

        self.x[:, 3] = np.array([r * np.cos(np.pi / 6), -r * np.sin(np.pi / 6), 0])
        self.x[:, 5] = np.array([0, r, 0])
        self.x[:, 7] = np.array([-r * np.cos(np.pi / 6), -r * np.sin(np.pi / 6), 0])
        self.x[:, 4] = (self.x[:, 3] + self.x[:, 5]) / 2
        self.x[:, 6] = (self.x[:, 5] + self.x[:, 7]) / 2
        self.x[:, 8] = (self.x[:, 7] + self.x[:, 3]) / 2

        self.x[:, 9] = np.array([r * np.cos(np.pi / 6), -r * np.sin(np.pi / 6), self.w])
        self.x[:, 10] = np.array([0, r, self.w])
        self.x[:, 11] = np.array([-r * np.cos(np.pi / 6), -r * np.sin(np.pi / 6), self.w])


def forward_kinematics(b, b0, x):
    """
    Calculates the forward kinematics of the Golden Ratio Yoshimura structure.

    Parameters:
    - b (numpy.ndarray): Current vertex positions.
    - b0 (numpy.ndarray): Reference vertex positions.
    - x (numpy.ndarray): Vertex positions to transform.

    Returns:
    - numpy.ndarray: Transformed vertex positions.
    """
    ob, ob0 = b[:, 0], b0[:, 0]
    b = b - ob[:, np.newaxis]
    b0 = b0 - ob0[:, np.newaxis]

    r = R.align_vectors(b.T, b0.T)[0].as_matrix()
    x = np.matmul(r, x - ob0[:, np.newaxis]) + ob[:, np.newaxis]
    t = ob - np.matmul(r, ob0)

    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return x, T


def capture(event, scene=None, filename='capture'):
    """
    Captures the VPython scene as an image.

    Parameters:
    - event: Event triggering the capture.
    - scene (vpython.canvas, optional): VPython scene to capture. Default is None.
    - filename (str, optional): Name of the capture file. Default is 'capture'.
    """
    if scene is None:
        scene = vp.canvas.get_selected()
    scene.capture(filename)


def get_shape(ip_str, scene=None, base=None, halt=True):
    """
   Constructs a Yoshimura structure based on the input string representing the configuration.

   Parameters:
   - ip_str (str): Input string representing the configuration of the Yoshimura structure.
   - scene (vpython.canvas): VPython scene for visualization.
   - base (numpy.array): Base vertices of the structure.
   - halt (bool): Flag indicating whether to halt after constructing the structure.

   Returns:
   - shape (list): List of Yoshimura modules forming the structure.

   Configuration Spaces:
   '000': No pop-out modules.
   '001': One pop-out module at phase 2.
   '010': One pop-out module at phase 1.
   '011': Two pop-out modules at phases 0.
   '100': One pop-out module at phase 0.
   '101': Two pop-out modules at phases 1.
   '110': Two pop-out modules at phases 2.
   '111': Three pop-out modules at phase 0.

   Example:
   pops = ['000', '001', '010', '011', '100', '101', '110', '111']
   shape = get_shape(pops, scene=sc)
   """
    config_spaces = {
        '000': [ZeroPopOut, 0],
        '001': [OnePopOut, 2],
        '010': [OnePopOut, 1],
        '011': [TwoPopOut, 0],
        '100': [OnePopOut, 0],
        '101': [TwoPopOut, 1],
        '110': [TwoPopOut, 2],
        '111': [ThreePopOut, 0],
    }

    if base is None:
        l = 1 / 3 ** 0.5
        tri = np.array([[l * np.cos(np.pi / 6), -l * np.sin(np.pi / 6), 0],
                        [0, l, 0],
                        [-l * np.cos(np.pi / 6), -l * np.sin(np.pi / 6), 0]]).T
    else:
        tri = base

    shape = []
    obj, phase = config_spaces[ip_str[0]]
    shape.append(obj(base=tri, phase=phase, scene=scene))

    for i in ip_str[1:]:
        obj, phase = config_spaces[i]
        shape.append(obj(base=shape[-1].top, phase=phase, scene=scene))

    if scene is not None and halt:
        spin(shape, scene)
    else:
        return shape


def spin(shape, scene, loop=True):
    """
    Spin the Yoshimura structure in the VPython scene.

    Parameters:
    - shape (list): List of Yoshimura modules forming the structure.
    - scene (vpython.canvas): VPython scene for visualization.
    - loop (bool): Flag indicating whether to loop the animation continuously.

    Usage:
    Click the checkboxes to show/hide the shape and individual cells. Click the 'Capture' button to take a screenshot.
    """

    # show/hide shape and individual cells
    scene.append_to_caption('\n\n Hide Shape: ')
    vp.checkbox(bind=lambda s: [cell.show_shape() if not s.checked else cell.hide_shape() for cell in shape])

    scene.append_to_caption('  Hide Module: ')
    vp.checkbox(bind=lambda s: [cell.show_unit() if not s.checked else cell.hide_unit() for cell in shape])

    # capture button for taking screenshots
    scene.append_to_caption('  Capture: ')
    vp.button(text='Capture', bind=lambda: capture(None, scene))

    # loop the animation if flag is set to True
    while loop:
        vp.rate(100)


if __name__ == '__main__':
    """
    Instructions:
    
    1. Hide Shape: Use the checkboxes to toggle the visibility of the entire Yoshimura structure.
    2. Hide Cells: Use the checkboxes to toggle the visibility of individual Yoshimura cells.
    3. Capture: Click the 'Capture' button to take a screenshot of the current scene.
    
    To interact with the scene:
    - Use your mouse to rotate the view.
    - Use the scroll wheel to zoom in and out.
    - Click and drag with the right mouse button to pan.
    
    You can modify the 'pops' list in the code to change the configuration of the Yoshimura structure.
    """

    # Create a VPython scene
    sc = vp.canvas(title='Golden Yoshimura Kinematics', width=1000, height=1000, x=0, y=0,
                   center=vp.vector(0, 0, 0), background=vp.color.white)
    sc.ambient = vp.color.white * 0.4

    # Define the configurations for the Yoshimura structure
    pops = ['000', '001', '010', '011', '100', '101', '110', '111']

    # Construct the Yoshimura structure
    shape = get_shape(pops, scene=sc)

    # Spin the structure and provide user instructions
    spin(shape, sc)
