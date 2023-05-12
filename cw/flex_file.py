"""
Load in object hierarchies from files based on their file extension.

"""


from collections.abc import Sequence
from pathlib import Path, PurePath
from typing import Union, Any, Tuple
import os
import json
import pickle
import gzip

from cw.serializers import mat, yaml, msgpack  # , xlsx


binary_mode_table = {
    pickle: True,
    yaml: False,
    msgpack: True,
    json: False,
    mat: True,
    # xlsx: True
}
"""
Dictionary indicating whether the serializers require the file to be opened in binary mode or not.
"""

extension_table = {
    ".pickle": pickle,
    ".yaml": yaml,
    ".yml": yaml,
    ".msgp": msgpack,
    ".json": json,
    ".mat": mat,
    # ".xlsx": xlsx
}
"""
Dictionary mapping file extensions to serializers.
"""

# TODO: Finish xlsx implementation.


def determine_serialization(file_path: Union[os.PathLike, str, PurePath],
                            default_serializer=None,
                            default_is_gzipped=False) -> Tuple[Any, bool, bool]:
    """
    Determines which serializer needs to be used to read or save a file based on its
    extension(s). Supports `.pickle`, `.yaml`, `.yml`, `.msgp`, `.json` and `.mat`. An
    optional additional `.gz` extension indicates the file is compressed using gzip.

    :param file_path: Path to file to open.
    :param default_serializer: Default serializer to use if the file extension is not known.
    :param default_is_gzipped: True to gzip if the file extension is not known.
    :return: Tuple with the serializer object, a boolean indicating whether the file is gzipped
             and a boolean indicating whether the file needs to be opened in binary mode.
    """

    file_path = Path(file_path)

    # Get file extensions.
    file_extensions = file_path.suffixes

    # Check whether the file is gzipped or not.
    if ".gz" in file_extensions:
        is_gzipped = True
        file_extensions.remove(".gz")
    else:
        is_gzipped = False

    # Get the extension indicating the serializer. If the file is gzipped, it's the
    # one before last extension. If the file is't gzipped it's the last one.
    serializer_extension = file_extensions[-1]

    # Check the file extension.
    serializer = extension_table.get(serializer_extension, None)
    if serializer is None:
        # The file extension was not recognized. Use the first serializer in the serializers list.

        # We'll be using a default serializer, so let's set is_gzipped to is_gzipped_default if
        # is_gzipped is False
        if not is_gzipped:
            is_gzipped = default_is_gzipped

        if default_serializer is None:
            # If no default serializers where given as a parameter, use pickle.
            serializer = pickle
        else:
            # If a single default serializer was given, just use that one.
            serializer = default_serializer

    # Check if the serializer is binary.
    is_binary = binary_mode_table[serializer]

    return serializer, is_gzipped, is_binary


def flex_load(file_path: Union[str, os.PathLike, PurePath],
              default_serializer=None,
              default_is_gzipped=False) -> Union[dict, list]:
    """
    Determines which serializer is needed to open the file and whether it's compressed by
    looking at the file extension. Supports `.pickle`, `.yaml`, `.yml`, `.msgp`, `.json`
    and `.mat`. An optional additional `.gz` extension indicates the file is compressed
    using gzip.

    :param file_path: Path to the file to load.
    :param default_serializer: Default serializer to use if the extension is unknown.
    :param default_is_gzipped: True if a file with unknown extension is compressed.
    :return: Object hierarchy stored in the file.
    """

    # Convert path to Path object and find the serializer, whether it's compressed and
    # whether the serialization is binary.
    file_path = Path(file_path)
    serializer, is_gzipped, is_binary = determine_serialization(
        file_path,
        default_serializer,
        default_is_gzipped
    )

    # Open the file and load data.
    open_mode = "rb" if is_binary or is_gzipped else "r"
    with file_path.open(open_mode) as f:
        # Read raw data
        raw_data = f.read()

    # Decompress if it's gzipped
    if is_gzipped:
        raw_data = gzip.decompress(raw_data)

    # Convert to bytes or string if necessary.
    if is_binary:
        raw_data = bytes(raw_data, "utf8") if isinstance(raw_data, str) else raw_data
    else:
        raw_data = str(raw_data, "utf8") if isinstance(raw_data, bytes) else raw_data

    return serializer.loads(raw_data)


def flex_dump(obj: Any,
              file_path: Union[str, os.PathLike, PurePath],
              default_serializer=None,
              default_is_gzipped=False,
              *args,
              **kwargs):
    """
    Dumps object hierarchies to a file. The serialization and compression is chosen based on the file extension.
    Supports `.pickle`, `.yaml`, `.yml`, `.msgp`, `.json` and `.mat`. An optional additional `.gz` extension
    indicates the file is compressed using gzip.

    :param obj: Object hierarchy to dump.
    :param file_path: Path to the file to dump the data to.
    :param default_serializer: Default serializer to use if the extension is unknown.
    :param default_is_gzipped: True if a file with unknown extension is compressed.
    :return:
    """
    file_path = Path(file_path)
    serializer, is_gzipped, is_binary = determine_serialization(
        file_path,
        default_serializer,
        default_is_gzipped
    )
    # print(serializer, file_path.suffixes, is_gzipped, is_binary)
    raw_data = serializer.dumps(obj, *args, **kwargs)

    if is_gzipped:
        raw_data = bytes(raw_data, "utf8") if isinstance(raw_data, str) else raw_data
        raw_data = gzip.compress(raw_data)

    open_mode = "wb" if is_binary or is_gzipped else "w"
    with file_path.open(open_mode) as f:
        f.write(raw_data)



