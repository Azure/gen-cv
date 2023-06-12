


field_json = {
    "name": "",
    "type": "Edm.String",
    "searchable": True,
    "filterable": True,
    "retrievable": True,
    "sortable": True,
    "facetable": True,
    "key": False,
    "indexAnalyzer": None,
    "searchAnalyzer": None,
    "analyzer": None,
    "normalizer": None,
    "dimensions": None,
    "vectorSearchConfiguration": None,
    "synonymMaps": []
}


vector_json = {
    "name": "vector",
    "type": "Collection(Edm.Single)",
    "searchable": True,
    "filterable": False,
    "retrievable": True,
    "sortable": False,
    "facetable": False,
    "key": False,
    "indexAnalyzer": None,
    "searchAnalyzer": None,
    "analyzer": None,
    "normalizer": None,
    "dimensions": 1536,
    "vectorSearchConfiguration": "vector-config",
    "synonymMaps": []
}


create_index_json = {
    "@odata.context": "https://cogvecseearch.search.windows.net/$metadata#indexes/$entity",
    "@odata.etag": "\"0x8DB40C97F04622D\"",
    "name": "vec-index",
    "defaultScoringProfile": None,
    "fields": [
        {
            "name": "id",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,
            "retrievable": True,
            "sortable": True,
            "facetable": True,
            "key": True,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "normalizer": None,
            "dimensions": None,
            "vectorSearchConfiguration": None,
            "synonymMaps": []
        },
        {
            "name": "text",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,
            "retrievable": True,
            "sortable": True,
            "facetable": True,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "normalizer": None,
            "dimensions": None,
            "vectorSearchConfiguration": None,
            "synonymMaps": []
        },
        {
            "name": "class",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,
            "retrievable": True,
            "sortable": True,
            "facetable": True,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "normalizer": None,
            "dimensions": None,
            "vectorSearchConfiguration": None,
            "synonymMaps": []
        },
        {
            "name": "file",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,
            "retrievable": True,
            "sortable": True,
            "facetable": True,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "normalizer": None,
            "dimensions": None,
            "vectorSearchConfiguration": None,
            "synonymMaps": []
        },
        {
            "name": "text_en",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,
            "retrievable": True,
            "sortable": True,
            "facetable": True,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "normalizer": None,
            "dimensions": None,
            "vectorSearchConfiguration": None,
            "synonymMaps": []
        },
        {
            "name": "categoryId",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,
            "retrievable": True,
            "sortable": True,
            "facetable": True,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "normalizer": None,
            "dimensions": None,
            "vectorSearchConfiguration": None,
            "synonymMaps": []
        },
        {
            "name": "aoi_text_vector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "normalizer": None,
            "dimensions": 1536,
            "vectorSearchConfiguration": "vector-config",
            "synonymMaps": []
        },
        {
            "name": 'cv_image_vector',
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "normalizer": None,
            "dimensions": 1024,
            "vectorSearchConfiguration": "vector-config",
            "synonymMaps": []
        },
        {
            "name": 'cv_text_vector',
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "filterable": False,
            "retrievable": True,
            "sortable": False,
            "facetable": False,
            "key": False,
            "indexAnalyzer": None,
            "searchAnalyzer": None,
            "analyzer": None,
            "normalizer": None,
            "dimensions": 1024,
            "vectorSearchConfiguration": "vector-config",
            "synonymMaps": []
        }
    ],
    "scoringProfiles": [],
    "corsOptions": {
        "allowedOrigins": [
            "*"
        ],
        "maxAgeInSeconds": 60
    },
    "suggesters": [],
    "analyzers": [],
    "normalizers": [],
    "tokenizers": [],
    "tokenFilters": [],
    "charFilters": [],
    "encryptionKey": None,
    "similarity": {
        "@odata.type": "#Microsoft.Azure.Search.BM25Similarity",
        "k1": None,
        "b": None
    },
    "semantic": {
        "defaultConfiguration": None,
        "configurations": [
            {
                "name": "semantic-config",
                "prioritizedFields": {
                    "prioritizedContentFields": [
                        {
                            "fieldName": "text_en"
                        }
                    ],
                    "prioritizedKeywordsFields": [
                        {
                            "fieldName": "categoryId"
                        }
                    ]
                }
            }
        ]
    },
    "vectorSearch": {
        "algorithmConfigurations": [
            {
                "name": "vector-config",
                "kind": "hnsw",
                "hnswParameters": {
                    "m": 10,
                    "efConstruction": 400,
                    "metric": "cosine"
                }
            }
        ]
    }
}


upload_doc_json = {
    "id": "",
    "text": "",
    "text_en": "",
    "file": "",
    "class": "",
    "categoryId": "",
    "aoi_text_vector": [],
    "@search.action": "upload"
}

upload_docs_json = {
    "value": [
    ]
}



search_dict_vector = {
    "vector": {
        "value": [],
        "fields": "aoi_text_vector",
        "k": 5
    },
    "select": "*",
    "filter": None
}


search_dict_hybrid = {
    "vector": {
        "value": [],
        "fields": "aoi_text_vector",
        "k": 5
    },
    "search": "",
    "select": "*",
    "top": f"{5}",
    "filter": None
}


search_dict_semantic_hybrid= {
    "vector": {
        "value": [],
        "fields": "aoi_text_vector",
        "k": 5,
    },
    "search": "",
    "select":"*",
    "queryType": "semantic",
    "semanticConfiguration": "semantic-config",
    "queryLanguage": "en-us",
    "captions": "extractive",
    "answers": "extractive",
    "top": 5,
    "filter": None
}

