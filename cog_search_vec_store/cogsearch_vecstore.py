import uuid
import os
import logging
import json
import copy
import re

from cog_search_vec_store import http_helpers
from cog_search_vec_store import cs_json
from cog_search_vec_store import cv_helpers

from utils import get_embedding, get_cosine_similarity, get_text_embedding
from utils import get_openai_embedding, analyze_image, save_obj_to_pkl


NUM_TOP_MATCHES = 5


class CogSearchVecStore:

    def __init__(self, api_key, 
                       search_service_name, 
                       index_name = "img-vec-index", 
                       api_version = "2023-07-01-Preview"):


        self.http_req = http_helpers.CogSearchHttpRequest(api_key, search_service_name, index_name, api_version)
        self.index_name = index_name
        self.all_fields = ['id', 'text', 'text_en', 'categoryId', 'file', 'class']
        self.search_types = ['vector', 'hybrid', 'semantic_hybrid']



    def create_index(self):
        
        index_dict = copy.deepcopy(cs_json.create_index_json)
        index_dict['name'] = self.index_name

        self.http_req.put(body = index_dict)


    def get_index(self):
        return self.http_req.get()


    def delete_index(self):
        return self.http_req.delete()


    def upload_documents(self, documents):

        docs_dict = copy.deepcopy(cs_json.upload_docs_json)

        for doc in documents:
            doc_dict = copy.deepcopy(cs_json.upload_doc_json)
                        
            for k in self.all_fields:
                doc_dict[k] = doc.get(k, '')

            doc_dict['id'] = doc['id'] if doc.get('id', None) else str(uuid.uuid4())
            doc_dict["aoi_text_vector"] = doc.get("aoi_text_vector", [])
            doc_dict['cv_image_vector'] = doc.get('cv_image_vector', [])
            doc_dict['cv_text_vector'] = doc.get('cv_text_vector', [])
            doc_dict["@search.action"] = "upload"
            docs_dict['value'].append(doc_dict)
        
        self.http_req.post(op ='index', body = docs_dict)

        return docs_dict



    def delete_documents(self, op='index', ids = []):
        docs_dict = copy.deepcopy(cs_json.upload_docs_json)

        for i in ids:
            doc_dict = copy.deepcopy(cs_json.upload_doc_json)
            doc_dict['id'] = i
            doc_dict["aoi_text_vector"] = [0] * 1536 ## text-embedding-ada-002 dimensions
            doc_dict["@search.action"] = "delete"
            docs_dict['value'].append(doc_dict)

        self.http_req.post(op ='index', body = docs_dict)



    def get_search_json(self, query, search_type = 'vector'):
        if search_type == 'vector':
            query_dict = copy.deepcopy(cs_json.search_dict_vector)
        elif search_type == 'hybrid':
            query_dict = copy.deepcopy(cs_json.search_dict_hybrid)
            query_dict['search'] = query
        elif search_type == 'semantic_hybrid':
            query_dict = copy.deepcopy(cs_json.search_dict_semantic_hybrid)
            query_dict['search'] = query
        return query_dict

            
    def get_vector_fields(self, query, query_dict, vector_name = None):
        if (vector_name is None) or (vector_name == "aoi_text_vector"):
            query_dict['vector']['fields'] = "aoi_text_vector"
            query_dict['vector']['value'] = get_openai_embedding(query, 'text-embedding-ada-002')    
        elif vector_name == 'cv_text_vector':
            cvr = cv_helpers.CV()
            query_dict['vector']['fields'] = vector_name
            query_dict['vector']['value'] = cvr.get_text_embedding(query)
        elif vector_name == 'cv_image_vector':
            cvr = cv_helpers.CV()
            query_dict['vector']['fields'] = vector_name
            query_dict['vector']['value'] = cvr.get_img_embedding(query)
        else:
            raise Exception(f'Invalid Vector Name {vector_name}')
        
        return query_dict



    def search(self, query, search_type = 'vector', vector_name = None, select=None, filter=None, verbose=False):
        analysis = ''

        if search_type not in self.search_types:
            raise Exception(f"search_type must be one of {self.search_types}")

        regex = r"(https?:\/\/[^\/\s]+(?:\/[^\/\s]+)*\/[^?\/\s]+(?:\.jpg|\.jpeg|\.png)(?:\?[^\s'\"]+)?)"
        match = re.search(regex, query)

        if match:
            inp_url = match.group(1)
            cvr = cv_helpers.CV()
            analysis = cvr.analyze_image(img_url=inp_url)
            query = query.replace(inp_url, '') + '\n' + analysis['text']

        query_dict = self.get_search_json(query, search_type)
        query_dict = self.get_vector_fields(query, query_dict, vector_name)
        query_dict['vector']['k'] = f"{NUM_TOP_MATCHES}"
        query_dict['filter'] = filter
        query_dict['select'] = ', '.join(self.all_fields) if select is None else select

        results = self.http_req.post(op ='search', body = query_dict)
        results = results['value'][:NUM_TOP_MATCHES]
        if verbose: [print(r['@search.score']) for r in results]
        if verbose: print(results)

        context, links, scores = self.process_search_results(results)

        if match:
            return ['Analysis of the image in the question: ' + query + '\n\n'] + context, links, scores, analysis
        else:
            return context, links, scores, analysis



    def search_similar_images(self, query, analyze = False, select=None, filter=None, verbose=False):

        analysis = ''
        search_type = 'vector'
        vector_name = 'cv_image_vector'

        if search_type not in self.search_types:
            raise Exception(f"search_type must be one of {self.search_types}")

        regex = r"(https?:\/\/[^\/\s]+(?:\/[^\/\s]+)*\/[^?\/\s]+(?:\.jpg|\.jpeg|\.png)(?:\?[^\s'\"]+)?)"
        match = re.search(regex, query)

        if match:
            url = match.group(1)
            query_dict = self.get_search_json(url, search_type)
            query_dict = self.get_vector_fields(url, query_dict, vector_name)
            if analyze: 
                cvr = cv_helpers.CV()
                analysis = cvr.analyze_image(img_url=url)
            query_dict['vector']['k'] = NUM_TOP_MATCHES
            query_dict['filter'] = filter
            query_dict['select'] = ', '.join(self.all_fields) if select is None else select

            results = self.http_req.post(op ='search', body = query_dict)
            results = results['value'][:NUM_TOP_MATCHES]
            if verbose: [print(r['@search.score']) for r in results]

            context, links, scores = self.process_search_results(results)

            return context, links, scores, analysis
        
        else:
            return ["Sorry, no similar images have been found"], [], [], analysis




        
    def process_search_results(self, results):

        if len(results) == 0:
            return ["Sorry, I couldn't find any information related to the question."]

        context = []
        links = []
        scores = []

        for t in results:
            t['text_en'] = t['text_en'].replace('\r', ' ').replace('\n', ' ') 
            links.append(t['file'])
            scores.append(t['@search.score'])

            try:
                if ('file' in t.keys()) and (t['file'] is not None) and (t['file'] != ''):
                    context.append('######\n' + f"[{t['file']}] " + t['text_en'] + '\n######\n')
                else:
                    context.append('######\n' + f"[{t['container']}/{t['filename']}] " + t['text_en']  + '\n######\n')
            except Exception as e:
                print("------------------- Exception in process_search_results: ", e)
                context.append('######\n' + f"[{t['file']}] " + t['text_en'] + '\n######\n')
                # context.append('######\n' + t['text_en'] + '\n######\n')

        final_context = []
        total_tokens = 0

        for i in range(len(context)):
            final_context.append(context[i])

        return final_context, links, scores