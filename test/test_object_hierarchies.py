import unittest
from cw.object_hierarchies import object_hierarchy_equals, object_hierarchy_to_tables
from cw.object_hierarchies.to_table import process_ndarray
from cw.object_hierarchies.from_table import flatten_tables, find_ndarrays, process_ndarrays
from decimal import Decimal
import numpy as np
import numpy.testing as npt
from pprint import PrettyPrinter
from pathlib import PurePosixPath


pprint = PrettyPrinter(indent=4).pprint


class TestObjectHierarchyCompare(unittest.TestCase):
    def test_equals_lists(self):
        """
        Simple tests that checks two lists with elements of different numerical types.
        """
        l1 = [0, 1., 2., 3, 4, Decimal(5)]
        l2 = [0, 1, 2., (3j*-1j), Decimal(4), (5+0j)]

        errors = object_hierarchy_equals(l1, l2)
        self.assertEqual(len(errors), 0)

    def test_equals_scalars_failure(self):
        """"""
        errors = object_hierarchy_equals(1, 2)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], "Value mismatch at '/'")

    def test_equals_dictionary(self):
        """
        Test that checks two trees with dictionaries as the root element is checked
        correctly. No errors should be found during this test.
        """
        d1 = {
            "copper": "explain",
            "truck": 2,
            "neat": [0, 1, 2., (3j*-1j), Decimal(4)],
            "unite": {
                "branch": "educated",
                "tenuous": 4.,
                "hum": {7, 8, 9},
                "decisive": np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
            },
            "notice": [
                "cattle",
                2.,
                5.4j,
                {"a", "b", "c", "c"},
                {
                    "team": "sneeze",
                    "warn": "deadpan",
                    "table": np.array([1, 2, 3, 4])
                }
            ]

        }

        d2 = {
            "copper": "explain",
            "truck": 2.,
            "neat": [0, 1., 2, Decimal(3), 4],
            "unite": {
                "branch": "educated",
                "tenuous": 4,
                "hum": {7, 8, 9, 8},
                "decisive": np.array([[7., 8., 9.], [4, 5, 6], [1, 2, 3]])
            },
            "notice": [
                "cattle",
                2.,
                5.4j,
                {"a", "b", "c"},
                {
                    "team": "sneeze",
                    "warn": "deadpan",
                    "table": [1, 2, 3, 4]
                }
            ]
        }

        errors = object_hierarchy_equals(d1, d2)
        # print("\n".join(errors))
        self.assertEqual(len(errors), 0)

    def test_equals_dictionary_failure(self):
        """
        Test that checks two trees with dictionaries as the root element is checked
        correctly.
        """
        d1 = {
            "copper": "limit",
            "truck": 3,
            "neat": [0, 0, 2., (3j*1j), Decimal(4)],
            "unite": {
                "branch": "educated",
                "tenuous": 4.2,
                "hum": {7, 8, 9, 10},
                "decisive": np.array([[7, 8, 9.1], [4, 5, 6], [1, 2.4, 3]])
            },
            "automatic": [
                "cattle",
                2.,
                5.4j,
                {"a", "b", "c"},
                {
                    "team": "sneeze",
                    "warn": "deadpan",
                    "table": np.array([1, 2, 3, 4])
                }
            ]

        }

        d2 = {
            "copper": "explain",
            "truck": 2.,
            "neat": [0, 1., 2, Decimal(3), 4],
            "unite": {
                "branch": "educated",
                "tenuous": 4,
                "hum": {7, 8, 9, 8},
                "decisive": [[7., 8., 9.], [4, 5, 6], [1, 2, 3]]
            },
            "notice": [
                "cattle",
                2.,
                5.4j,
                {"a", "b", "c"},
                {
                    "team": "sneeze",
                    "warn": "deadpan",
                    "table": [1, 2, 3, 4]
                }
            ]
        }

        errors = object_hierarchy_equals(d1, d2)

        expected_errors = [
            "Missing element '/notice' in the first object hierarchy.",
            "Value mismatch at '/copper'",
            "Value mismatch at '/truck'",
            "Value mismatch at '/neat[1]'",
            "Value mismatch at '/neat[3]'",
            "Value mismatch at '/unite/tenuous'",
            "Value mismatch at '/unite/hum'",
            "Value mismatch at '/unite/decisive[0][2]'",
            "Value mismatch at '/unite/decisive[2][1]'",
            "Missing element '/automatic' in the second object hierarchy."
        ]

        # print("\n".join(errors))

        self.assertEqual(len(errors), len(expected_errors))
        for expected_error in expected_errors:
            with self.subTest(expected_error=expected_error):
                self.assertIn(expected_error, errors)

    def test_object_hierarchy_to_tables(self):
        """
        Test to see if the object_hierarchy_to_tables works properly.
        """
        inp = {
            "foo": 0,
            "bar": [1, 2],
            "baz": np.array([3, 4, 5]),
            "ham": {
                "qux": np.array([6, 7, 8]),
                "quux": np.array([[[9],  [10], [11]],
                                  [[11], [12], [13]],
                                  [[14], [15], [16]],
                                  [[17], [18], [19]]]),
            },
            "spam": [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        }

        correct = {
            0: {'foo': 0},
            2: {'bar': [1, 2]},
            3: {'baz': np.array([3, 4, 5]),
                'ham.qux': np.array([6, 7, 8]),
                'spam.[0]': [1, 2, 3],
                'spam.[1]': [4, 5, 6],
                'spam.[2]': [7, 8, 9]},
            4: {'ham.quux_0_0': np.array([9, 11, 14, 17]),
                'ham.quux_1_0': np.array([10, 12, 15, 18]),
                'ham.quux_2_0': np.array([11, 13, 16, 19])}}

        out = object_hierarchy_to_tables(inp)
        # pprint(out)

        errors = object_hierarchy_equals(out, correct)
        self.assertEqual(len(errors), 0)

    def test_object_hierarchy_to_tables_process_ndarray(self):
        # FIRST DIMENSION SHOULD BE EQUAL FOR ALL DATA ARRAYS
        a1 = np.arange(0, 10 ** 3).reshape((10, 1, 1, 1, 10, 10, 1))
        a2 = np.arange(0, 30).reshape((10, 3, 1))
        a3 = np.arange(0, 10)

        a1_correct = [
            ["name_0_0_0_0_0_0", [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]],
            ["name_0_0_0_0_1_0", [1, 101, 201, 301, 401, 501, 601, 701, 801, 901]],
            ["name_0_0_0_0_2_0", [2, 102, 202, 302, 402, 502, 602, 702, 802, 902]],
            ["name_0_0_0_0_3_0", [3, 103, 203, 303, 403, 503, 603, 703, 803, 903]],
            ["name_0_0_0_0_4_0", [4, 104, 204, 304, 404, 504, 604, 704, 804, 904]],
            ["name_0_0_0_0_5_0", [5, 105, 205, 305, 405, 505, 605, 705, 805, 905]],
            ["name_0_0_0_0_6_0", [6, 106, 206, 306, 406, 506, 606, 706, 806, 906]],
            ["name_0_0_0_0_7_0", [7, 107, 207, 307, 407, 507, 607, 707, 807, 907]],
            ["name_0_0_0_0_8_0", [8, 108, 208, 308, 408, 508, 608, 708, 808, 908]],
            ["name_0_0_0_0_9_0", [9, 109, 209, 309, 409, 509, 609, 709, 809, 909]],
            ["name_0_0_0_1_0_0", [10, 110, 210, 310, 410, 510, 610, 710, 810, 910]],
            ["name_0_0_0_1_1_0", [11, 111, 211, 311, 411, 511, 611, 711, 811, 911]],
            ["name_0_0_0_1_2_0", [12, 112, 212, 312, 412, 512, 612, 712, 812, 912]],
            ["name_0_0_0_1_3_0", [13, 113, 213, 313, 413, 513, 613, 713, 813, 913]],
            ["name_0_0_0_1_4_0", [14, 114, 214, 314, 414, 514, 614, 714, 814, 914]],
            ["name_0_0_0_1_5_0", [15, 115, 215, 315, 415, 515, 615, 715, 815, 915]],
            ["name_0_0_0_1_6_0", [16, 116, 216, 316, 416, 516, 616, 716, 816, 916]],
            ["name_0_0_0_1_7_0", [17, 117, 217, 317, 417, 517, 617, 717, 817, 917]],
            ["name_0_0_0_1_8_0", [18, 118, 218, 318, 418, 518, 618, 718, 818, 918]],
            ["name_0_0_0_1_9_0", [19, 119, 219, 319, 419, 519, 619, 719, 819, 919]],
            ["name_0_0_0_2_0_0", [20, 120, 220, 320, 420, 520, 620, 720, 820, 920]],
            ["name_0_0_0_2_1_0", [21, 121, 221, 321, 421, 521, 621, 721, 821, 921]],
            ["name_0_0_0_2_2_0", [22, 122, 222, 322, 422, 522, 622, 722, 822, 922]],
            ["name_0_0_0_2_3_0", [23, 123, 223, 323, 423, 523, 623, 723, 823, 923]],
            ["name_0_0_0_2_4_0", [24, 124, 224, 324, 424, 524, 624, 724, 824, 924]],
            ["name_0_0_0_2_5_0", [25, 125, 225, 325, 425, 525, 625, 725, 825, 925]],
            ["name_0_0_0_2_6_0", [26, 126, 226, 326, 426, 526, 626, 726, 826, 926]],
            ["name_0_0_0_2_7_0", [27, 127, 227, 327, 427, 527, 627, 727, 827, 927]],
            ["name_0_0_0_2_8_0", [28, 128, 228, 328, 428, 528, 628, 728, 828, 928]],
            ["name_0_0_0_2_9_0", [29, 129, 229, 329, 429, 529, 629, 729, 829, 929]],
            ["name_0_0_0_3_0_0", [30, 130, 230, 330, 430, 530, 630, 730, 830, 930]],
            ["name_0_0_0_3_1_0", [31, 131, 231, 331, 431, 531, 631, 731, 831, 931]],
            ["name_0_0_0_3_2_0", [32, 132, 232, 332, 432, 532, 632, 732, 832, 932]],
            ["name_0_0_0_3_3_0", [33, 133, 233, 333, 433, 533, 633, 733, 833, 933]],
            ["name_0_0_0_3_4_0", [34, 134, 234, 334, 434, 534, 634, 734, 834, 934]],
            ["name_0_0_0_3_5_0", [35, 135, 235, 335, 435, 535, 635, 735, 835, 935]],
            ["name_0_0_0_3_6_0", [36, 136, 236, 336, 436, 536, 636, 736, 836, 936]],
            ["name_0_0_0_3_7_0", [37, 137, 237, 337, 437, 537, 637, 737, 837, 937]],
            ["name_0_0_0_3_8_0", [38, 138, 238, 338, 438, 538, 638, 738, 838, 938]],
            ["name_0_0_0_3_9_0", [39, 139, 239, 339, 439, 539, 639, 739, 839, 939]],
            ["name_0_0_0_4_0_0", [40, 140, 240, 340, 440, 540, 640, 740, 840, 940]],
            ["name_0_0_0_4_1_0", [41, 141, 241, 341, 441, 541, 641, 741, 841, 941]],
            ["name_0_0_0_4_2_0", [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]],
            ["name_0_0_0_4_3_0", [43, 143, 243, 343, 443, 543, 643, 743, 843, 943]],
            ["name_0_0_0_4_4_0", [44, 144, 244, 344, 444, 544, 644, 744, 844, 944]],
            ["name_0_0_0_4_5_0", [45, 145, 245, 345, 445, 545, 645, 745, 845, 945]],
            ["name_0_0_0_4_6_0", [46, 146, 246, 346, 446, 546, 646, 746, 846, 946]],
            ["name_0_0_0_4_7_0", [47, 147, 247, 347, 447, 547, 647, 747, 847, 947]],
            ["name_0_0_0_4_8_0", [48, 148, 248, 348, 448, 548, 648, 748, 848, 948]],
            ["name_0_0_0_4_9_0", [49, 149, 249, 349, 449, 549, 649, 749, 849, 949]],
            ["name_0_0_0_5_0_0", [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]],
            ["name_0_0_0_5_1_0", [51, 151, 251, 351, 451, 551, 651, 751, 851, 951]],
            ["name_0_0_0_5_2_0", [52, 152, 252, 352, 452, 552, 652, 752, 852, 952]],
            ["name_0_0_0_5_3_0", [53, 153, 253, 353, 453, 553, 653, 753, 853, 953]],
            ["name_0_0_0_5_4_0", [54, 154, 254, 354, 454, 554, 654, 754, 854, 954]],
            ["name_0_0_0_5_5_0", [55, 155, 255, 355, 455, 555, 655, 755, 855, 955]],
            ["name_0_0_0_5_6_0", [56, 156, 256, 356, 456, 556, 656, 756, 856, 956]],
            ["name_0_0_0_5_7_0", [57, 157, 257, 357, 457, 557, 657, 757, 857, 957]],
            ["name_0_0_0_5_8_0", [58, 158, 258, 358, 458, 558, 658, 758, 858, 958]],
            ["name_0_0_0_5_9_0", [59, 159, 259, 359, 459, 559, 659, 759, 859, 959]],
            ["name_0_0_0_6_0_0", [60, 160, 260, 360, 460, 560, 660, 760, 860, 960]],
            ["name_0_0_0_6_1_0", [61, 161, 261, 361, 461, 561, 661, 761, 861, 961]],
            ["name_0_0_0_6_2_0", [62, 162, 262, 362, 462, 562, 662, 762, 862, 962]],
            ["name_0_0_0_6_3_0", [63, 163, 263, 363, 463, 563, 663, 763, 863, 963]],
            ["name_0_0_0_6_4_0", [64, 164, 264, 364, 464, 564, 664, 764, 864, 964]],
            ["name_0_0_0_6_5_0", [65, 165, 265, 365, 465, 565, 665, 765, 865, 965]],
            ["name_0_0_0_6_6_0", [66, 166, 266, 366, 466, 566, 666, 766, 866, 966]],
            ["name_0_0_0_6_7_0", [67, 167, 267, 367, 467, 567, 667, 767, 867, 967]],
            ["name_0_0_0_6_8_0", [68, 168, 268, 368, 468, 568, 668, 768, 868, 968]],
            ["name_0_0_0_6_9_0", [69, 169, 269, 369, 469, 569, 669, 769, 869, 969]],
            ["name_0_0_0_7_0_0", [70, 170, 270, 370, 470, 570, 670, 770, 870, 970]],
            ["name_0_0_0_7_1_0", [71, 171, 271, 371, 471, 571, 671, 771, 871, 971]],
            ["name_0_0_0_7_2_0", [72, 172, 272, 372, 472, 572, 672, 772, 872, 972]],
            ["name_0_0_0_7_3_0", [73, 173, 273, 373, 473, 573, 673, 773, 873, 973]],
            ["name_0_0_0_7_4_0", [74, 174, 274, 374, 474, 574, 674, 774, 874, 974]],
            ["name_0_0_0_7_5_0", [75, 175, 275, 375, 475, 575, 675, 775, 875, 975]],
            ["name_0_0_0_7_6_0", [76, 176, 276, 376, 476, 576, 676, 776, 876, 976]],
            ["name_0_0_0_7_7_0", [77, 177, 277, 377, 477, 577, 677, 777, 877, 977]],
            ["name_0_0_0_7_8_0", [78, 178, 278, 378, 478, 578, 678, 778, 878, 978]],
            ["name_0_0_0_7_9_0", [79, 179, 279, 379, 479, 579, 679, 779, 879, 979]],
            ["name_0_0_0_8_0_0", [80, 180, 280, 380, 480, 580, 680, 780, 880, 980]],
            ["name_0_0_0_8_1_0", [81, 181, 281, 381, 481, 581, 681, 781, 881, 981]],
            ["name_0_0_0_8_2_0", [82, 182, 282, 382, 482, 582, 682, 782, 882, 982]],
            ["name_0_0_0_8_3_0", [83, 183, 283, 383, 483, 583, 683, 783, 883, 983]],
            ["name_0_0_0_8_4_0", [84, 184, 284, 384, 484, 584, 684, 784, 884, 984]],
            ["name_0_0_0_8_5_0", [85, 185, 285, 385, 485, 585, 685, 785, 885, 985]],
            ["name_0_0_0_8_6_0", [86, 186, 286, 386, 486, 586, 686, 786, 886, 986]],
            ["name_0_0_0_8_7_0", [87, 187, 287, 387, 487, 587, 687, 787, 887, 987]],
            ["name_0_0_0_8_8_0", [88, 188, 288, 388, 488, 588, 688, 788, 888, 988]],
            ["name_0_0_0_8_9_0", [89, 189, 289, 389, 489, 589, 689, 789, 889, 989]],
            ["name_0_0_0_9_0_0", [90, 190, 290, 390, 490, 590, 690, 790, 890, 990]],
            ["name_0_0_0_9_1_0", [91, 191, 291, 391, 491, 591, 691, 791, 891, 991]],
            ["name_0_0_0_9_2_0", [92, 192, 292, 392, 492, 592, 692, 792, 892, 992]],
            ["name_0_0_0_9_3_0", [93, 193, 293, 393, 493, 593, 693, 793, 893, 993]],
            ["name_0_0_0_9_4_0", [94, 194, 294, 394, 494, 594, 694, 794, 894, 994]],
            ["name_0_0_0_9_5_0", [95, 195, 295, 395, 495, 595, 695, 795, 895, 995]],
            ["name_0_0_0_9_6_0", [96, 196, 296, 396, 496, 596, 696, 796, 896, 996]],
            ["name_0_0_0_9_7_0", [97, 197, 297, 397, 497, 597, 697, 797, 897, 997]],
            ["name_0_0_0_9_8_0", [98, 198, 298, 398, 498, 598, 698, 798, 898, 998]],
            ["name_0_0_0_9_9_0", [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]]
        ]

        a2_correct = [
            ["name_0_0", [0,  3,  6,  9, 12, 15 ,18, 21, 24, 27]],
            ["name_1_0", [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]],
            ["name_2_0", [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]],
        ]

        a3_correct = [["name", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]]

        sub_tests = [
            ['a1', a1, a1_correct],
            ['a2', a2, a2_correct],
            ['a3', a3, a3_correct],
        ]

        for test_name, inp, correct_list in sub_tests:
            with self.subTest(name=test_name):
                for (values, path), (name, correct_values) in zip(process_ndarray(inp, PurePosixPath("name")), correct_list):
                    npt.assert_allclose(values, correct_values)
                    self.assertEqual(str(path), name)

    def test_tables_to_object_hierarchy_flatten_tables(self):
        data = {
            0: {'foo': 0},
            2: {'bar': [1, 2]},
            3: {'baz': np.array([3, 4, 5]),
                'ham.qux': np.array([6, 7, 8]),
                'spam.[0]': [1, 2, 3],
                'spam.[1]': [4, 5, 6],
                'spam.[2]': [7, 8, 9]},
            "eggs": {'ham.quux_0_0': np.array([9, 11, 14, 17]),
                     'ham.quux_1_0': np.array([10, 12, 15, 18]),
                     'ham.quux_2_0': np.array([11, 13, 16, 19])}}

        pprint(flatten_tables(data))

    def test_tables_to_object_hierarchy_find_ndarrays(self):
        data = flatten_tables({
            0: {'foo': 0},
            2: {'bar': [1, 2]},
            3: {'baz': np.array([3, 4, 5]),
                'ham.qux': np.array([6, 7, 8]),
                'spam.[0]': [1, 2, 3],
                'spam.[1]': [4, 5, 6],
                'spam.[2]': [7, 8, 9],
                'quux_0_0': np.array([9, 11, 14, 17]),
                'quux_1_0': np.array([10, 12, 15, 18]),
                'quux_2_0': np.array([11, 13, 16, 19])
                },
            "eggs": {'ham.quux_0_0': np.array([9, 11, 14, 17]),
                     'ham.quux_1_0': np.array([10, 12, 15, 18]),
                     'ham.quux_2_0': np.array([11, 13, 16, 19])
                     }})

        correct_result = {
            PurePosixPath('eggs/ham/quux'): [
                (PurePosixPath('eggs/ham/quux_0_0'),
                 (0, 0)),
                (PurePosixPath('eggs/ham/quux_1_0'),
                 (1, 0)),
                (PurePosixPath('eggs/ham/quux_2_0'),
                 (2, 0))],
            PurePosixPath('quux'): [
                (PurePosixPath('quux_0_0'),
                 (0, 0)),
                (PurePosixPath('quux_1_0'),
                 (1, 0)),
                (PurePosixPath('quux_2_0'),
                 (2, 0))]}

        results = {}
        for path, group in find_ndarrays(data):
            results[path] = []
            for data in group:
                results[path].append(data)

        errors = object_hierarchy_equals(results, correct_result)
        self.assertEqual(len(errors), 0)

    def test_process_ndarrays(self):
        data = flatten_tables({
            0: {'foo': 0},
            2: {'bar': [1, 2]},
            3: {'baz': np.array([3, 4, 5]),
                'ham.qux': np.array([6, 7, 8]),
                'spam.[0]': [1, 2, 3],
                'spam.[1]': [4, 5, 6],
                'spam.[2]': [7, 8, 9],
                'quux_0_0': np.array([9, 11, 14, 17]),
                'quux_1_0': np.array([10, 12, 15, 18]),
                'quux_2_0': np.array([11, 13, 16, 19])
                },
            "eggs": {'ham.quux_0_0': np.array([9, 11, 14, 17]),
                     'ham.quux_1_0': np.array([10, 12, 15, 18]),
                     'ham.quux_2_0': np.array([11, 13, 16, 19])
                     }})

        process_ndarrays(data)

        correct = {
            PurePosixPath('bar'): [1, 2],
            PurePosixPath('baz'): np.array([3, 4, 5]),
            PurePosixPath('eggs/ham/quux'):
                np.array([[[9.],  [10.], [11.]],
                          [[11.], [12.], [13.]],
                          [[14.], [15.], [16.]],
                          [[17.], [18.], [19.]]]),
            PurePosixPath('foo'): 0,
            PurePosixPath('ham/qux'): np.array([6, 7, 8]),
            PurePosixPath('quux'):
                np.array([[[9.],  [10.], [11.]],
                          [[11.], [12.], [13.]],
                          [[14.], [15.], [16.]],
                          [[17.], [18.], [19.]]]),
            PurePosixPath('spam/[0]'): [1, 2, 3],
            PurePosixPath('spam/[1]'): [4, 5, 6],
            PurePosixPath('spam/[2]'): [7, 8, 9]}

        errors = object_hierarchy_equals(data, correct)
        self.assertEqual(len(errors), 0)
















































