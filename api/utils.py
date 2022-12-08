import hashlib
import copy
import os


def allowed_file(filename):
    """
    Checks if the format for the file received is acceptable. For this
    particular case, we must accept only image files. This is, files with
    extension ".png", ".jpg", ".jpeg" or ".gif".

    Parameters
    ----------
    filename : str
        Filename from werkzeug.datastructures.FileStorage file.

    Returns
    -------
    bool
        True if the file is an image, False otherwise.
    """
    # Current implementation will allow any kind of file.
    # TODO
    state = filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
    return state


def get_file_hash(file, file_name):
    """
    Returns a new filename based on the file content using MD5 hashing.
    It uses hashlib.md5() function from Python standard library to get
    the hash.

    Parameters
    ----------
    file : werkzeug.datastructures.FileStorage
        File sent by user.

    Returns
    -------
    str
        New filename based in md5 file hash.
    """
    # Current implementation will return the original file name.
    # TODO

    #contents = file.stream.read()
    #filename_content = hashlib.md5(contents)
    file_copy = copy.deepcopy(file)
    content = file_copy.read()
    filename_content = hashlib.md5(content)
    name_file = filename_content.hexdigest() + "." + file_name.split('.')[1]
    return str(name_file)