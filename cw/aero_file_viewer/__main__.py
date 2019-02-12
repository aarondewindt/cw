"""
makes a 1D/2D/3D (slice)plot of a certain coefficient from an aero file
requires user input for the location of the aero file, coefficient choice and parameter ranges.

example command: apy -m sim_common.aero_file_viewer -i sim_common/aero_file_viewer/aero_1.yaml -c c_x -p ' -5:6:0.1' ' 0.4'
"""
from cw.aero_file_viewer.aero_file_viewer import *
import argparse
from cw.aero_file import AeroFile

arg_parser = argparse.ArgumentParser("Aero file viewer", "Program used to plot a coefficient from an aero file")
arg_parser.add_argument("-i", "--input_file", help="Location of the aero file")
arg_parser.add_argument("-c", "--coefficient", help="The name of the coefficient you want to plot")
arg_parser.add_argument("-p", "--paramRanges", nargs='+', help="Parameter ranges, ' start1:stop1:step1, ' start2:stop2:step2,....")
args = arg_parser.parse_args()

if args.input_file is None:
    print("please give file to open:")
    path = input()
else:
    path = args.input_file

aero = AeroFile(path) #open aero file

if args.coefficient is None:
    coefficientName = getCoefficient(aero)
else:
    coefficientName = args.coefficient

parameters = []
nonConstantParameters = [0]
getParameters(aero, coefficientName, parameters)

if args.paramRanges is not None:
    putScope(args.paramRanges, parameters, nonConstantParameters)
else:
    getScope(parameters, nonConstantParameters)

fillAxis(parameters)

if nonConstantParameters[0] == 0:
    plot1d(aero, coefficientName, parameters)

elif nonConstantParameters[0] == 1:
    plot2d(aero, coefficientName, parameters)

elif nonConstantParameters[0] == 2:
    plot3d(aero, coefficientName, parameters)

elif nonConstantParameters[0] > 2:
    print("too many dimentions")
