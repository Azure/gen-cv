import requests
import json

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


# """
# api_key = 'YOUR_API_KEY'
# search_service_name = 'your-search-service-name'
# index_name = 'your-index-name'
# api_version = 'your-api-version'

# request = HTTPRequest(api_key, search_service_name, index_name, api_version)
# headers = {'Authorization': 'Bearer YOUR_ACCESS_TOKEN'}
# body = {'key': 'value'}

# response_put = request.put(headers=headers, body=body)
# response_post = request.post(headers=headers, body=body)
# response_get = request.get(headers=headers)
# response_delete = request.delete(headers=headers)

# print(response_put)
# print(response_post)
# print(response_get)
# print(response_delete)
# """


class HTTPError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP Error {status_code}: {message}")



class HTTPRequest:
    def __init__(self, url = '', api_key = ''):
        self.url = url
        self.api_key = api_key
        self.default_headers = {'Content-Type': 'application/json', 'api-key': self.api_key}
        
        
    def initialize_for_cogsearch(self, api_key, search_service_name, index_name, api_version):
        self.api_key = api_key
        self.search_service_name = search_service_name
        self.index_name = index_name
        self.api_version = api_version
        self.url        = f"{search_service_name}/indexes/{index_name}?api-version={api_version}"
        self.post_url   = f"{search_service_name}/indexes/{index_name}/docs/index?api-version={api_version}"
        self.search_url = f"{search_service_name}/indexes/{index_name}/docs/search?api-version={self.api_version}"
        
        self.default_headers = {'Content-Type': 'application/json', 'api-key': self.api_key}


    def handle_response(self, response):
        try:
            response_data = json.loads(response.text)
        except json.JSONDecodeError:
            response_data = response.text

        if response.status_code >= 400:
            raise HTTPError(response.status_code, response_data)

        return response_data


    def get_url(self, op = None):
        return self.url


    @retry(wait=wait_random_exponential(min=1, max=4), stop=stop_after_attempt(4))
    def put(self, op = None, headers=None, body=None):
        
        url = self.get_url(op)

        if headers is None:
            headers = self.default_headers
        else:
            headers = {**self.default_headers, **headers}
        
        if body is None:
            body = {}
        
        response = requests.put(url, json=body, headers=headers)
        return self.handle_response(response)


    @retry(wait=wait_random_exponential(min=1, max=4), stop=stop_after_attempt(4))
    def post(self, op = None, headers=None, body=None, data=None):

        url = self.get_url(op)

        if headers is None:
            headers = self.default_headers
        else:
            headers = {**self.default_headers, **headers}
        
        if body is None:
            body = {}
        
        if data is not None:
            response = requests.post(url, data=data, headers=headers)
        elif body is not None:
            response = requests.post(url, json=body, headers=headers)
        else:
            response = requests.post(url, headers=headers)

        return self.handle_response(response)


    @retry(wait=wait_random_exponential(min=1, max=4), stop=stop_after_attempt(2))
    def get(self, op = None, headers=None, params=None):

        url = self.get_url(op)

        if headers is None:
            headers = self.default_headers
        else:
            headers = {**self.default_headers, **headers}
        
        if params is None:
            params = {}
        
        response = requests.get(url, headers=headers, params=params)
        return self.handle_response(response)


    @retry(wait=wait_random_exponential(min=1, max=4), stop=stop_after_attempt(4))
    def delete(self, op = None, id = None, headers=None):

        url = self.get_url(op)

        if headers is None:
            headers = self.default_headers
        else:
            headers = {**self.default_headers, **headers}
        
        response = requests.delete(url, headers=headers)
        return self.handle_response(response)





class CogSearchHttpRequest(HTTPRequest):

    def __init__(self, api_key, search_service_name, index_name, api_version):
        self.api_key = api_key
        self.search_service_name = search_service_name
        self.index_name = index_name
        self.api_version = api_version
        self.url        = f"{search_service_name}/indexes/{index_name}?api-version={api_version}"
        self.post_url   = f"{search_service_name}/indexes/{index_name}/docs/index?api-version={api_version}"
        self.search_url = f"{search_service_name}/indexes/{index_name}/docs/search?api-version={self.api_version}"
        
        self.default_headers = {'Content-Type': 'application/json', 'api-key': self.api_key}


    def get_url(self, op = None):
        if op == 'index':
            url = self.post_url
        elif op == 'search':
            url = self.search_url
        else:
            url = self.url

        return url



class CVHttpRequest(HTTPRequest):

    def __init__(self, api_key, cog_serv_name, api_version, 
                 options = ['tags', 'objects', 'caption', 'read', 'smartCrops', 'denseCaptions', 'people']):

        self.api_key = api_key

        if cog_serv_name.endswith('/'):
            cog_serv_name = cog_serv_name[:-1]

        self.cog_serv_name = cog_serv_name
        self.api_version = api_version

        options = ','.join(options).replace(' ', '') if isinstance(options, list) else options
        self.url = f"{cog_serv_name}/computervision/imageanalysis:analyze?api-version={api_version}&modelVersion=latest&features={options}"
        self.imgvec_url = f"{cog_serv_name}/computervision/retrieval:vectorizeImage?api-version={api_version}&modelVersion=latest"
        self.txtvec_url = f"{cog_serv_name}/computervision/retrieval:vectorizeText?api-version={api_version}&modelVersion=latest"
        
        self.default_headers = {'Content-type': 'application/octet-stream','Ocp-Apim-Subscription-Key': self.api_key}
        self.json_headers = {'Content-type': 'application/json','Ocp-Apim-Subscription-Key': self.api_key}


    def get_url(self, op = None):
        if op == 'analyze':
            url = self.url
        elif op == 'img_embedding':
            url = self.imgvec_url
        elif op == 'text_embedding':
            url = self.txtvec_url            
        else:
            url = self.url

        return url





