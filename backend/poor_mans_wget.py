import os
import requests


def wget(url, filename):
    ckpt_request = requests.get(url)
    request_status = ckpt_request.status_code

    # inform user of errors
    if request_status == 403:
        raise ConnectionRefusedError("You have not accepted the license for this model.")
    elif request_status == 404:
        raise ConnectionError("Could not make contact with server")
    elif request_status != 200:
        raise ConnectionError(f"Some other error has ocurred - response code: {request_status}")

    # write to model path
    with open(filename, 'wb') as model_file:
        model_file.write(ckpt_request.content)


def wget_headers(url):
    r = requests.get(url, stream=True, headers={'Connection':'close'})
    return r.headers

def wget_progress(url, filename, length=0, chunk_size=8192, callback=None):

    one_percent = int(length) / 100
    next_percent = 1

    with requests.get(url, stream=True) as r:

        r.raise_for_status()
        downloaded_bytes = 0
        callback(next_percent)
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
                downloaded_bytes += chunk_size
                if downloaded_bytes > next_percent * one_percent:
                    next_percent += 1
                    callback(next_percent)
