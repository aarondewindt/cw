# cw
This is a collection of python libraries I've written throughout the 
years, mostly during my time at Delft Aerospace Rocket Engineering 
(DARE) and studies.

Package | Description
--- | ---
aero_file | Class used to store, evaluate and (de)serialize aerodynamic models
aero_file_viewer | Tool used to visualize `cw.aero_file.AeroFile` instances
atmo | Atmospheric models and utilities
context | Collection of context managers, `chdir`, `profile_it`, `time_it`, `suppress_stdout`
control | Controllers and control theory utilities
fdlti | Linear flight models
fdm | Stub. Planned Flight Dynamics Model
filters | Signal filters and smoothers. Incl. Iterated Extended Kalman Filter (IEKF)
mp | Deprecated. Multiprocessing library
object_hierachies | Operations on hierachies of dictionaries and lists, aka,  JSON like data
serializers | (Wrappers around) (de)serializers with the same interface as `pickle`
simulation | Modular simulation library capable of handling continues and discrete modules
test | Unittests
tile_coding | Classes used for tile coding
vdom | Virtual DOM. Easily generate HTML, SVG and other XML using python and no text templates
xsens | Xsens sensor log parser
 | |
async_test | Deprecated. Decorator to allow for `async def` tests to be defined in a `unittest.TestCase`
cached | Decorator for creating cached properties
cli_base | Base class used to define a Command Line Interface. Used by `cw` CLI tools.
constants | Useful constants, eg. `g_earth`
conversions | Conversion functions and factors. Transform between reference frames and units
directory_checksum | Scans a directory and it subdirectories and creates a file with the checksum of all files in the directory
directory_walk | Iterates through all files in a directory and its subdirectories
downsample | Downsample data series
enable_notebook_import | Enable importing from Jupyter notebooks
event | `asyncio` event class
exceptions_decorators | Log or print exceptions happening in a (async) function
flex_file | Load and dump data to `.pickle`, `.yaml`, `.yml`, `.msgp`, `.json` and `.mat`. Add `.gz` at the end to gzip the contents
generate_cython_pyi | Not recommended. Generate a python stub file (`.pyi`) from a cython source file (`.pyx`). Only guarenteed to work with the two `.pyx` files I've used this with.
generate_paper_name | Generate the name of a PDF for a scientific paper with my preferred format
itertools | Iteration tools. `iterify`, `grouper`, `chunks`, `until`
jinja2_python_block | Jinja2 extension for embedding python in the templates
lbp | DARE LaunchBox Protocol implementation. Simple datagram based serial communication protocol for low bandwidth, low reliability serial connections, eg. PC <-> embedded systems
lerp | Linear interpolation between two points
net_tools | Networking tools. `get_available_port()`, `has_internet()`
numpy_monkey_patch | Allows you to create a numpy array by calling `np[[1, 2], [3, 4]]`
rm | Delete files, directories and symlinks, using the same function
shave | Shave outliers from a dataseries. Replaces them by interpolated values
singletons | Declare singleton type. Types of which there can only exists one instance
special_print | Special print functions. `debug_print`, `code_print`, `verbose_print`, `yaml_print`
synchronization | Multithreading synchronization classes. `BinarySemaphore`, `CheckInSemaphore`
tidy_source | Strips and fixes the indentation of python code
transformations | Creates transformation functions for transforming between flight dynamics reference frames.
tree_node | Tree structure. Able to search for nodes and do other stuff
version | `cw` version
wind_log | Logarithmic wind model. For modeling wind speeds up to 300m above ground level
