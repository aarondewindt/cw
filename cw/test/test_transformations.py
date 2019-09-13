import unittest
from cw.transformations import *


class Transformations(unittest.TestCase):
    def test_tr_ci(self):
        dcm = tr_ci(0, 0)
        self.assertEqual(np.linalg.det(dcm), 1)
    
    def test_tr_ic(self):
        dcm = tr_ic(0, 0)
        self.assertEqual(np.linalg.det(dcm), 1)

    def test_tr_ab(self):
        dcm = tr_ab(0, 0)
        self.assertEqual(np.linalg.det(dcm), 1)
        
    def test_tr_ba(self):
        dcm = tr_ba(0, 0)
        self.assertEqual(np.linalg.det(dcm), 1)
        
    def test_tr_ec(self):
        dcm = tr_ec(0, 0)
        self.assertEqual(np.linalg.det(dcm), 1)
        
    def test_tr_ce(self):
        dcm = tr_ce(0, 0)
        self.assertEqual(np.linalg.det(dcm), 1)
    
    def test_tr_be(self):
        dcm = tr_be(0, 0, 0)
        self.assertEqual(np.linalg.det(dcm), 1)
        
    def test_tr_eb(self):
        dcm = tr_eb(0, 0, 0)
        self.assertEqual(np.linalg.det(dcm), 1)
        
    def test_tr_ae(self):
        dcm = tr_ae(0, 0, 0)
        self.assertEqual(np.linalg.det(dcm), 1)
        
    def test_tr_ea(self):
        dcm = tr_ea(0, 0, 0)
        self.assertEqual(np.linalg.det(dcm), 1)