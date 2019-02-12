from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

# I think this was written by Jort Groen.


class Parameter:
    count = 0

    def __init__(self, name):
        self.count += 1
        self.name = name
        self.scope = np.zeros(3)
        self.axis = []
        self.X = False
        self.Y = False

# Parameter.count

def getCoefficient(aero):

    print("Current file contains the following coefficients:")
    for coefficientName, c in aero.coefficients.items():
        print("Name:", coefficientName, "\t model:", aero.coefficients[coefficientName].model_name, "\t(",
              len(aero.coefficients[coefficientName].parameter_names), "parameter(s))")
    print()
    print("Which coefficient do you want to plot?")
    coefficientName = input()
    return coefficientName

def getParameters(aero, coefficientName, parameters):
    """
    Finds all parameters inside the given coeficient and puts them into a list
    :param aero: The aero file
    :param coefficientName: The name of the coefficient
    :param parameters: List where the found parameters will be stored.
    """
    for par in aero.coefficients[coefficientName].parameter_names:
        new_par = Parameter(par)
        new_par.name = par
        parameters.append(new_par)

def putScope(ranges, parameters, nonConstantParameters):
    """
    Splits the ranges and puts them at the corresponding parameter
    :param ranges: the ranges as gotten via command
    :param parameters: List of parameters
    :param nonConstantParameters: The amount of non constant parameters
    :return:
    """
    if len(parameters) != len(ranges):
        print("error, invalid amount of ranges given.",len(ranges),"are given, expected",len(parameters))
        exit()

    i=0
    for parameter in parameters:
        parameter.scope = (list(map(float, ranges[i].split(":"))))
        if len(parameter.scope)==3:
            nonConstantParameters[0]+=1
        i+=1

def getScope(parameters, nonConstantParameters):
    """
    Asks user to define the ranges for the parameters.
    The parameters can be constant aswell, in that case, the user will only define one value.
    :param parameters: List containing all parameters, the ranges will be added to each parameter inside this list
    :param nonConstantParameters: Amount of non constant parameters.
    :return:
    """
    found = False
    for parameter in parameters:
        print("please provide a range and resolution for ", parameter.name, " (min:max:step):")

        scope = []
        while len(scope) != 1 and len(scope) != 3: #the number of arguments can only be either 1 or 3
            string = input()
            try:
                scope = (list(map(float, string.split(":"))))
            except:
                pass
            if len(scope) != 1 and len(scope) != 3:
                print("invalid amount of arguments")

        parameter.scope = scope
        if len(parameter.scope) == 1:
            print("parameter will be kept constant at:", parameter.scope[0])
        else:
            nonConstantParameters[0] += 1
            found = True

    # all parameters are constant
    # if found == False:
    #     for parameter in parameters:
    #         parameter.axis[0] = parameter.scope[0]


def fillAxis(parameters):
    """
    Creates an axis for each parameter. this axis will be from start to stop with the given step.
    If the parameter is a constant, the axis will contain only one value.
    All axis will be the same length (the length of the longest axis)
    :param parameters: List containing all parameters, the corresponding axis will be added to each parameter element
    :return:
    """
    longest = 0
    #fill axis of non-constant parameters
    for parameter in parameters:
        if len(parameter.scope) == 3: #non constant axis
            parameter.axis = np.arange(parameter.scope[0], parameter.scope[1], parameter.scope[2])
            if len(parameter.axis)>longest:
                longest = len(parameter.axis) #max amount of elements needed

    #fill axis of constant parameters
    for parameter in parameters:
        if len(parameter.scope) == 1:
            parameter.axis = np.empty(longest)
            parameter.axis.fill(parameter.scope[0])

def plot1d(aero,name,parameters):
    """
    Prints the coefficient value corresponding to the given parameters.
    :param aero: The aero file.
    :param name: The name of the coefficient.
    :param parameters: List of parameters.
    :return:
    """
    #get values
    values = [parameter.scope[0] for parameter in parameters]
    print("The corresponding value is: ",aero.coefficients[name].get_coefficient(values))  # get corresponding coefficient

def plot2d(aero,name,parameters):
    """
    Creates a 2D plot of the coefficient values corresponding to the given parameters.
    In the case of any constant parameters this plot represents a slice of a higher dimentional plot.
    :param aero: The aero file.
    :param name: The name of the coefficient.
    :param parameters: List of parameters.
    :return:
    """
    #find non-constant parameter
    for parameter in parameters:
        if(len(parameter.scope)==3): #this is our first axis
            X = parameter.axis
            xlabel = parameter.name

    Y = np.zeros(X.shape[0])

    i=0
    while i<X.shape[0]:
        values = [parameter.axis[i] for parameter in parameters] #get the i'th value of all parameters
        Y[i] = aero.coefficients[name].get_coefficient(*values) #get corresponding coefficient
        i+=1

    plt.plot(X,Y)
    plt.xlabel(xlabel)
    plt.ylabel(name)
    plt.show()


def plot3d(aero, name, parameters):
    """
    Creates a 3D plot of the coefficient values corresponding to the given parameters.
    In the case of any constant parameters this plot represents a slice of a higher dimentional plot.
    :param aero: The aero file.
    :param name: The name of the coefficient.
    :param parameters: List of parameters.
    :return:
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #find the X and Y axis (the two non constant axis)
    found=0
    for parameter in parameters:
        if(len(parameter.scope)==3): #if not constant
            if found==0:
                X=parameter.axis
                parameter.X = True
                found+=1
            if found==1:
                Y=parameter.axis
                parameter.Y = True
            else:
                print("error, too many non constant variables")

    Z = np.zeros((len(Y), len(X)))
    X, Y = np.meshgrid(X, Y)

    i = 0
    j = 0
    while i < X.shape[1]:
        j = 0
        while j < Y.shape[0]:
            # prepare list of values
            values = []
            for parameter in parameters:
                if parameter.X == True: #if its the X axis
                    values.append(X[0,i])
                    xlabel = parameter.name
                elif parameter.Y == True: #if its the Y axis
                    values.append(Y[j,0])
                    ylabel=parameter.name
                else:
                    values.append(parameter.axis[i])  # get the i'th value of all parameters

            Z[j, i] = aero.coefficients[name].get_coefficient(*values)
            j += 1
        i += 1

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, vmin=np.nanmin(Z), vmax=np.nanmax(Z),
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(np.nanmin(Z), np.nanmax(Z))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.xaxis.set_label_text(xlabel)
    ax.yaxis.set_label_text(ylabel)
    ax.zaxis.set_label_text(name)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()