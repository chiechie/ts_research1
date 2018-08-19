# encoding=utf-8
from os.path import dirname, join, exists, isdir, split, isfile
from os import makedirs, errno, listdir, remove, walk, mkdir
import json
try:
    import cPickle as pkl
except ImportError:
    import _pickle as pkl
import shutil
import hashlib

import pandas as pd


def load_data(file, episode):
    data = loadPklfrom(file)
    return map(list, zip(*[iter(data)] * episode))



def mkdir_p(path):
    try:
        makedirs(path)
        print("[mkdir_p]", path)
    except OSError as exc:
        # Python >2.5 (except OSError, exc: for Python <2.5)
        if exc.errno == errno.EEXIST and isdir(path):
            pass
        else:
            raise


def saveCSV(d, des_dir):
    file_path = get_normal_path(des_dir)
    if not file_path.startswith("/"):
        file_path = "/" + "/".join(file_path.split("/")[1:])
    mkdir_p(dirname(file_path))
    print("[saveCSV]%s" % des_dir)
    with open(des_dir, "w") as f:
        f.write(d)
    pass


def readCSV(des_dir):
    file_path = get_normal_path(des_dir)
    if not file_path.startswith("/"):
        file_path = "/" + "/".join(file_path.split("/")[1:])
    print("[readCSV]%s" % des_dir)
    with open(file_path, "r") as f:
        d = f.read()
    return d


def clean_dir(des_dir):
    if isfile(des_dir):
        remove(des_dir)
        print("[CleanPath][%s]" % des_dir)
        return
    des_dir_files = listdir(des_dir)
    if len(des_dir_files) > 0:
        try:
            shutil.rmtree(des_dir)
            mkdir(des_dir)
            print("[CleanPath][%s]" % des_dir)
            return
        except:
            return

def clean_path(class_csv_path, file_list):
    """
    删除某个文件夹下，没有在file_list出现的文件.csv
    :param class_csv_path:
    :param file_list:
    :return:
    """
    for file in listdir(class_csv_path):
        if "checkpoint" in file:
            continue
        save = False
        for f in file_list:
            if f in file:
                save = True
                break
        if not save:
            remove(join(class_csv_path, file))
            print("[remove]", join(class_csv_path, file))
    return


def moveFileto(sourceDir,  targetDir):
    if not exists(sourceDir):
        print("not exist", sourceDir)
        return sourceDir
    if exists(targetDir):
        return
    mkdir_p(dirname(targetDir))
    shutil.copy(sourceDir,  targetDir)
    return


def savePNG(plt, targetDir, **kwargs):
    if targetDir is None:
        return
    file_path = get_normal_path(targetDir)
    if not file_path.startswith("/"):
        file_path = "/" + "/".join(file_path.split("/")[1:])
    mkdir_p(dirname(file_path))
    print("[savePNG]%s" % file_path)
    plt.savefig(file_path, **kwargs)
    return


def savePklto(py_obj, targetDir):
    targetDir = get_normal_path(targetDir)
    mkdir_p(dirname(targetDir))
    print("[savePklto]%s" % targetDir)
    with open(targetDir, "wb") as f:
        pkl.dump(py_obj, f)
    return


def pywalker(path):
    file_absolute = []
    for root, dirs, files in walk(path):
        for file_ in files:
            file_absolute.append(join(root, file_) )
    return file_absolute


def loadPklfrom(sourceDir):
    print("[loadPklfrom]%s" % sourceDir)
    with open(sourceDir, "rb") as f:
        py_obj = pkl.load(f)
    return py_obj


def get_normal_path(a):
    # a = str(a)

    a = repr(a)
    a = a.replace('\\', '').replace('x00', '').replace('\'', '').replace('\"', '')
    if not a.startswith("/"):
        a = "/" + "/".join(a.split("/")[1:])
    return a


def saveJSON(df, file_path):
    file_path = get_normal_path(file_path)
    if not file_path.startswith("/"):
        file_path = "/" + "/".join(file_path.split("/")[1:])
    mkdir_p(dirname(file_path))
    print("[SavingJSON]%s" % file_path)
    with open(file_path, "wb") as f:
        f.write(json.dumps(df))
    return


def readJSON(file_path):
    file_path = get_normal_path(file_path)
    print("[readJSON]%s" % file_path)
    with open(file_path, "rb") as f:
        df = f.read()
    return json.loads(df)


def get_dir_file(dir, type=".json"):
    valid_lists = []
    for root, dirs, files in walk(dir):
        # valid_list = [i for i in files]
        valid_list = [join(root, i) for i in files if type in i]
        if len(valid_list) == 0:
            continue
        valid_lists.extend(valid_list)
    return valid_lists


def merge_df_list(df_list):
    cnt = 0
    if len(df_list) <= 0:
        return None
    for df in df_list:
        tmp = readDF(df)
        if cnt == 0:
            total_data = tmp
            cnt += 1
        else:
            total_data = pd.concat([total_data, tmp], axis=0)
            total_data = total_data.reset_index(drop=True)
        cnt += 1
    return total_data


def saveDF(df, file_path):
    file_path = get_normal_path(file_path)
    if not file_path.startswith("/"):
        file_path = "/" + "/".join(file_path.split("/")[1:])
    mkdir_p(dirname(file_path))
    print("[SavingDF]%s" % file_path)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, )
    df.to_csv(file_path, index=None)
    return


def readDF(file_path, **kwargs):
    file_path = get_normal_path(file_path)
    print("[readDF]%s" % file_path)
    df = pd.read_csv(file_path, **kwargs)
    return df


def split_dir(file_path):
    _dir, _filename = split(file_path)
    return _dir, _filename

def list_md5_string_value(list):
    string = json.dumps(list)
    return hashlib.md5(string).hexdigest()


# def setCache(k, v):
#     # N_mb = sys.getsizeof(v) / (1024 * 1024)
#     # if N_mb >= 1:
#     print("[setCache]%s" % k)
#     print(type(v))
#     set_cache(key=hash32(k), value=v, expires=86400 * 180,  use_external=False, use_disk=True)
#     return
#
#
# def getCache(k):
#     print "[getCache]%s" % k
#     v = get_cache(key=hash32(k), use_external=False, use_disk=True)
#     print(type(v))
#     return v


if __name__ == "__main__":
    pass