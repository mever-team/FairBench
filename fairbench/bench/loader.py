import pandas as pd
import zipfile
import os
import urllib.request


def _download(url, path=None):
    # Get the file name from the URL
    if path is None:
        file_name = os.path.basename(url)
    else:
        file_name = path

    try:
        with urllib.request.urlopen(url) as response:
            total_size = response.getheader("Content-Length")
            total_size = int(total_size) if total_size else None

            with open(file_name, "wb") as out_file:
                chunk_size = 1024
                downloaded = 0

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)

                    # Print progress if total size is known
                    if total_size:
                        done = int(50 * downloaded / total_size)
                        print(
                            f'\rDownloading {url} [{"=" * done}{" " * (50 - done)}] {downloaded / 1024:.2f} KB',
                            end="",
                        )

        print(f"Downloaded {url}" + " " * 50)
    except Exception as e:
        print(f"Error downloading file: {e}")


def _extract_nested_zip(file, folder):
    print(file, folder)
    os.makedirs(folder, exist_ok=True)
    with zipfile.ZipFile(file, "r") as zfile:
        zfile.extractall(path=folder)
    os.remove(file)
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".zip"):
                _extract_nested_zip(
                    os.path.join(root, filename), os.path.join(root, filename[:-4])
                )


def read_csv(url, *args, **kwargs):
    url = url.replace("\\", "/")
    if ".zip/" in url:
        url, path = url.split(".zip/", 1)
        extract_to = "data/"
        if "/" not in path:
            extract_to += url.split("/")[-1]
            path = os.path.join(url.split("/")[-1], path)
        path = os.path.join("data", path)
        url += ".zip"
        temp = "data/" + url.split("/")[-1]
        if not os.path.exists(path):
            os.makedirs(os.path.join(*path.split("/")[:-1]), exist_ok=True)
            _download(url, temp)
            _extract_nested_zip(temp, extract_to)
    else:
        shortened = "/".join(url.split("/")[-4:])
        path = "data/" + shortened
        if not os.path.exists(path):
            os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
            _download(url, path)
    return pd.read_csv(path, *args, **kwargs)


def features(df, numeric, categorical):
    dfs = [df[col] for col in numeric] + [
        pd.get_dummies(df[col]) for col in categorical
    ]
    return pd.concat(dfs, axis=1).values
