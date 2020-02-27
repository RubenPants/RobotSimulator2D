"""
myutils.py

Share utils used across the project.

Utils overview:
 * DICT: append_dict, load_dict, store_dict
 * JSON: append_json, load_json, store_json
 * SYSTEM: create_subfolder, python2_to_3
 * TIMING: drop, prep, status_out, total_time
"""
import glob
import json
import os
import sys
from timeit import default_timer as timer


# ------------------------------------------------------> DICT <------------------------------------------------------ #


def append_dict(full_path, new_dict):
    """
    Append existing dictionary if exists, otherwise create new.
    
    :param full_path: Path with name of JSON file (including '.json')
    :param new_dict: The JSON file that must be stored
    """
    files = glob.glob(full_path)
    
    if files:
        # Append new json
        with open(full_path, 'r') as f:
            original = json.load(f)
        
        for k in new_dict:
            original[k] += new_dict[k]
        
        store_json(new_json=original,
                   full_path=full_path,
                   indent=2)
    else:
        # Create new file to save json in
        store_json(new_json=new_dict,
                   full_path=full_path,
                   indent=2)


def clip_dict(full_path, i):
    """
    Clip the dictionary to only the first i elements, if the type of a key's value is a list.
    
    :param full_path: Path with name of JSON file (including '.json')
    :param i: Clipping index: only the first i elements in a list are considered
    """
    with open(full_path, 'r') as f:
        original = json.load(f)
    for k in original:
        v = original[k]
        if type(v) == list:
            original[k] = v[:i]
    store_json(new_json=original,
               full_path=full_path,
               indent=2)


def get_fancy_string_dict(d, title=None):
    """
    Return the string result of the fancy-print for a given dictionary.
    
    :param d: Dictionary
    :param title: [Optional] Title before print
    :return: String
    """
    s = title + '\n' if title else ''
    space = max(map(lambda x: len(x), d.keys()))
    for k, v in d.items():
        s += '\t{key:>{s}s} : {value}\n'.format(s=space, key=k, value=v)
    return s


def load_dict(full_path):
    """
    Load the dictionary stored under 'full_path'.
    
    :param full_path: Path with name of JSON file (including '.json')
    :return: Dictionary or FileNotFound (Exception)
    """
    return load_json(full_path)


def update_dict(full_path, new_dict):
    """
    Update existing dictionary if exists, otherwise create new.
    
    :param full_path: Path with name of JSON file (including '.json')
    :param new_dict: The JSON file that must be stored
    """
    files = glob.glob(full_path)
    
    if files:
        # Append new json
        with open(full_path, 'r') as f:
            original = json.load(f)
        
        original.update(new_dict)
        
        store_json(new_json=original,
                   full_path=full_path,
                   indent=2)
    else:
        # Create new file to save json in
        store_json(new_json=new_dict,
                   full_path=full_path,
                   indent=2)


# ------------------------------------------------------> JSON <------------------------------------------------------ #


def append_json(full_path, new_json):
    """
    Append existing JSON file if exists, otherwise create new json.
    
    :param full_path: Path with name of JSON file (including '.json')
    :param new_json: The JSON file that must be stored
    """
    files = glob.glob(full_path)
    
    if files:
        # Append new json
        with open(full_path, 'r') as f:
            original = json.load(f)
        
        original += new_json
        
        store_json(new_json=original,
                   full_path=full_path,
                   indent=2)
    else:
        # Create new file to save json in
        store_json(new_json=new_json,
                   full_path=full_path,
                   indent=2)


def load_json(full_path):
    """
    Load the JSON file stored under 'full_path'.
    
    :param full_path: Path with name of JSON file (including '.json')
    :return: JSON or FileNotFound (Exception)
    """
    with open(full_path, 'r') as f:
        return json.load(f)


def store_json(new_json, full_path, indent=2):
    """
    Write a JSON file, if one already exists under the same 'full_path' then this file will be overwritten.
    
    :param full_path: Path with name of JSON file (including '.json')
    :param new_json: The JSON file that must be stored
    :param indent: Indentation used to pretty-print JSON
    """
    with open(full_path, 'w') as f:
        json.dump(new_json, f, indent=indent)


# -----------------------------------------------------> SYSTEM <----------------------------------------------------- #

def get_subfolder(path, subfolder, init=True):
    """
    Check if subfolder already exists in given directory, if not, create one.
    
    :param path: Path in which subfolder should be located (String)
    :param subfolder: Name of the subfolder that must be created (String)
    :param init: Initialize folder with __init__.py file (Bool)
    :return: Path name if exists or possible to create, raise exception otherwise
    """
    if subfolder and subfolder[-1] not in ['/', '\\']:
        subfolder += '/'
    
    # Path exists
    if os.path.isdir(path) or path == '':
        if not os.path.isdir(path + subfolder):
            # Folder does not exist, create new one
            os.mkdir(path + subfolder)
            if init:
                with open(path + subfolder + '__init__.py', 'w') as f:
                    f.write('')
        return path + subfolder
    
    # Given path does not exist, raise Exception
    raise FileNotFoundError("Path '{p}' does not exist".format(p=path))


# -----------------------------------------------------> TIMING <----------------------------------------------------- #

time_dict = dict()


def drop(key=None, silent=False):
    """
    Stop timing, print out duration since last preparation and append total duration.
    """
    # Update dictionary
    global time_dict
    if key not in time_dict:
        raise Exception("prep() must be summon first")
    time_dict[key]['end'] = timer()
    time_dict[None]['end'] = timer()
    
    # Determine difference
    start = time_dict[key]['start']
    end = time_dict[key]['end']
    diff = end - start
    
    # Fancy print diff
    if not silent:
        status_out(' done, ' + get_fancy_time(diff) + '\n')
    
    # Save total time
    if key is not None:
        if 'sum' not in time_dict[key]:
            time_dict[key]['sum'] = diff
        else:
            time_dict[key]['sum'] += diff
    
    return diff


def get_fancy_time(sec):
    """
    Convert a time measured in seconds to a fancy-printed time.
    
    :param sec: Float
    :return: String
    """
    h = int(sec) // 3600
    m = (int(sec) // 60) % 60
    s = sec % 60
    if h > 0:
        return '{h} hours, {m} minutes, and {s} seconds.'.format(h=h, m=m, s=round(s, 2))
    elif m > 0:
        return '{m} minutes, and {s} seconds.'.format(m=m, s=round(s, 2))
    else:
        return '{s} seconds.'.format(s=round(s, 2))


def intermediate(key=None):
    """
    Get the time that already has passed.
    """
    # Update dictionary
    global time_dict
    if key not in time_dict:
        raise Exception("prep() must be summon first")
    
    # Determine difference
    start = time_dict[key]['start']
    end = timer()
    return end - start


def prep(msg="Start timing...", key=None, silent=False):
    """
    Prepare timing, print out the given message.
    """
    global time_dict
    if key not in time_dict:
        time_dict[key] = dict()
    if not silent:
        status_out(msg)
    time_dict[key]['start'] = timer()
    
    # Also create a None-instance (in case drop() is incorrect)
    if key:
        if None not in time_dict:
            time_dict[None] = dict()
        time_dict[None]['start'] = timer()


def print_all_stats():
    """
    Print out each key and its total (cumulative) time.
    """
    global time_dict
    if time_dict:
        if None in time_dict: del time_dict[None]  # Remove None-instance first
        print("\n\n\n---------> OVERVIEW OF CALCULATION TIME <---------\n")
        keys_space = max(map(lambda x: len(x), time_dict.keys()))
        line = ' {0:^' + str(keys_space) + 's} - {1:^s}'
        line = line.format('Keys', 'Total time')
        print(line)
        print("-" * (len(line) + 3))
        line = '>{0:^' + str(keys_space) + 's} - {1:^s}'
        t = 0
        for k, v in sorted(time_dict.items()):
            try:
                t += v['sum']
                print(line.format(k, get_fancy_time(v['sum'])))
            except KeyError:
                print("KeyError: Probably you forgot to place a 'drop()' in the {} section".format(k))
                raise KeyError
        end_line = line.format('Total time', get_fancy_time(t))
        print("-" * (len(end_line)))
        print(end_line)


def remove_time_key(key: str):
    """
    Remove a key from the timing-dictionary.
    
    :param key: Key that must be removed
    """
    global time_dict
    if key in time_dict:
        del time_dict[key]


def status_out(msg):
    """
    Write the given message.
    """
    sys.stdout.write(msg)
    sys.stdout.flush()
