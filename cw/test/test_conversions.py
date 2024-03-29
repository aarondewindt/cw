import unittest
from cw.conversions import *
from math import pi, isclose


class Conversions(unittest.TestCase): 
    def test_dcm_to_euler(self):
        input =  [[0.4741598818, 0.7026971887, 0.5304611837, 0.2590347240, 0.4644898356, -0.8468472143, -0.8414709848, 0.5389488414, 0.0382194732],
                  [-0.4446838280, -0.5068853259, 0.7384602626, 0.8937075973, -0.1963081039, 0.4034226801, -0.0595233027, 0.8393630887, 0.5403023059],
                  [-0.3403343662, -0.2590347240, 0.9039211973, 0.7705236308, 0.4741598818, 0.4259879587, -0.5389488414, 0.8414709848, 0.0382194732],
                  [-0.1963081039, -0.8937075973, 0.4034226801, 0.5068853259, -0.4446838280, -0.7384602626, 0.8393630887, 0.0595233027, 0.5403023059],
                  [0.4644898356, -0.8468472143, 0.2590347240, 0.5389488414, 0.0382194732, -0.8414709848, 0.7026971887, 0.5304611837, 0.4741598818],
                  [-0.1963081039, 0.4034226801, 0.8937075973, 0.8393630887, 0.5403023059, -0.0595233027, -0.5068853259, 0.7384602626, -0.4446838280],
                  [0.4741598818, 0.4259879587, 0.7705236308, 0.8414709848, 0.0382194732, -0.5389488414, -0.2590347240, 0.9039211973, -0.3403343662],
                  [-0.4446838280, -0.7384602626, 0.5068853259, 0.0595233027, 0.5403023059, 0.8393630887, -0.8937075973, 0.4034226801, -0.1963081039],
                  [0.0382194732, -0.5389488414, 0.8414709848, 0.9039211973, -0.3403343662, -0.2590347240, 0.4259879587, 0.7705236308, 0.4741598818],
                  [0.5403023059, 0.8393630887, 0.0595233027, 0.4034226801, -0.1963081039, -0.8937075973, -0.7384602626, 0.5068853259, -0.4446838280],
                  [0.0382194732, -0.8414709848, 0.5389488414, 0.5304611837, 0.4741598818, 0.7026971887, -0.8468472143, 0.2590347240, 0.4644898356],
                  [0.5403023059, -0.0595233027, 0.8393630887, 0.7384602626, -0.4446838280, -0.5068853259, 0.4034226801, 0.8937075973, -0.1963081039]]
        
        rtypes = ['zyx', 'zyz', 'zxy', 'zxz', 'yxz', 'yxy', 'yzx', 'yzy', 'xyz', 'xyx', 'xzy', 'xzx']
    
        for i, rtype in enumerate(rtypes):
            dcm = np.array(input[i]).reshape((3,3)).transpose()
            angle = dcm_to_euler(dcm, rtype)
    
            self.assertAlmostEqual(0.5, angle[0])
            self.assertAlmostEqual(1.0, angle[1])
            self.assertAlmostEqual(1.5, angle[2])

    def test_angle2DCM(self):
        results =  [[0.4741598818, 0.7026971887, 0.5304611837, 0.2590347240, 0.4644898356, -0.8468472143, -0.8414709848, 0.5389488414, 0.0382194732],
                    [-0.4446838280, -0.5068853259, 0.7384602626, 0.8937075973, -0.1963081039, 0.4034226801, -0.0595233027, 0.8393630887, 0.5403023059],
                    [-0.3403343662, -0.2590347240, 0.9039211973, 0.7705236308, 0.4741598818, 0.4259879587, -0.5389488414, 0.8414709848, 0.0382194732],
                    [-0.1963081039, -0.8937075973, 0.4034226801, 0.5068853259, -0.4446838280, -0.7384602626, 0.8393630887, 0.0595233027, 0.5403023059],
                    [0.4644898356, -0.8468472143, 0.2590347240, 0.5389488414, 0.0382194732, -0.8414709848, 0.7026971887, 0.5304611837, 0.4741598818],
                    [-0.1963081039, 0.4034226801, 0.8937075973, 0.8393630887, 0.5403023059, -0.0595233027, -0.5068853259, 0.7384602626, -0.4446838280],
                    [0.4741598818, 0.4259879587, 0.7705236308, 0.8414709848, 0.0382194732, -0.5389488414, -0.2590347240, 0.9039211973, -0.3403343662],
                    [-0.4446838280, -0.7384602626, 0.5068853259, 0.0595233027, 0.5403023059, 0.8393630887, -0.8937075973, 0.4034226801, -0.1963081039],
                    [0.0382194732, -0.5389488414, 0.8414709848, 0.9039211973, -0.3403343662, -0.2590347240, 0.4259879587, 0.7705236308, 0.4741598818],
                    [0.5403023059, 0.8393630887, 0.0595233027, 0.4034226801, -0.1963081039, -0.8937075973, -0.7384602626, 0.5068853259, -0.4446838280],
                    [0.0382194732, -0.8414709848, 0.5389488414, 0.5304611837, 0.4741598818, 0.7026971887, -0.8468472143, 0.2590347240, 0.4644898356],
                    [0.5403023059, -0.0595233027, 0.8393630887, 0.7384602626, -0.4446838280, -0.5068853259, 0.4034226801, 0.8937075973, -0.1963081039]]
    
        rtypes = ['zyx', 'zyz', 'zxy', 'zxz', 'yxz', 'yxy', 'yzx', 'yzy', 'xyz', 'xyx', 'xzy', 'xzx']
    
        for i, rtype in enumerate(rtypes):
            dcm = euler_to_dcm(0.5,1.,1.5, rtype, 'rad')
    
            self.assertAlmostEqual(dcm[0,0], results[i][0], msg='Falied at "%s".'%(rtype))
            self.assertAlmostEqual(dcm[1,0], results[i][1], msg='Falied at "%s".'%(rtype))
            self.assertAlmostEqual(dcm[2,0], results[i][2], msg='Falied at "%s".'%(rtype))
            self.assertAlmostEqual(dcm[0,1], results[i][3], msg='Falied at "%s".'%(rtype))
            self.assertAlmostEqual(dcm[1,1], results[i][4], msg='Falied at "%s".'%(rtype))
            self.assertAlmostEqual(dcm[2,1], results[i][5], msg='Falied at "%s".'%(rtype))
            self.assertAlmostEqual(dcm[0,2], results[i][6], msg='Falied at "%s".'%(rtype))
            self.assertAlmostEqual(dcm[1,2], results[i][7], msg='Falied at "%s".'%(rtype))
            self.assertAlmostEqual(dcm[2,2], results[i][8], msg='Falied at "%s".'%(rtype))

    def test_angle_to_rot_2d_and_rot_to_angle_2d(self):
        angles = [0, 0.25*pi, 0.5*pi, 0.75*pi, pi, -0.25*pi, -0.5*pi, -0.75*pi]
        for angle in angles:
            with self.subTest(angle=angle):
                rot = angle_to_rot_2d(angle)
                angle_2 = rot_to_angle_2d(rot)
                # print(angle, angle_2, np.isclose(angle, angle_2))
                self.assertTrue(isclose(angle, angle_2))



