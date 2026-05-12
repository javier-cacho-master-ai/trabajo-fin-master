import json
import sys
from urllib.parse import urlencode

import requests

def mast_query(request):
    """Perform a MAST query.

        Parameters
        ----------
        request (dictionary): The MAST request json object

        Returns head,content where head is the response HTTP headers, and content is the returned data"""

    # Base API url
    request_url = 'https://mast.stsci.edu/api/v0/invoke'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent": "python-requests/"+version}

    # Encoding the request as a json string
    req_string = json.dumps(request)
    req_string = urlencode({"request": req_string})

    # Perform the HTTP request
    resp = requests.post(request_url, data=req_string, headers=headers)

    # Pull out the headers and response content
    head = resp.headers
    content = resp.content.decode('utf-8')

    return head, content

def set_filters(parameters):
    return [{"paramName": p, "values": v} for p, v in parameters.items()]


def set_min_max(min, max):
    return [{'min': min, 'max': max}]
