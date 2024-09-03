# IMPORTANT NOTICE:
# THIS FILE REPLACES THE wget.download METHOD WITH A PURE PYTHON EQUIVALENT THAT SATISFIES THE NEEDS OF THIS PROJECT

import os
import urllib.request


def download(url, path=None):
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

        print(f"Downloaded {url}"+" "*50)
    except Exception as e:
        print(f"Error downloading file: {e}")
