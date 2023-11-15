import logging
import os
import json
from azure.communication.networktraversal import CommunicationRelayClient
from azure.communication.identity import CommunicationIdentityClient

import azure.functions as func

connection_str = os.getenv("ICE_CONNECTION_STRING")

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    identity_client = CommunicationIdentityClient.from_connection_string(connection_str)
    relay_client = CommunicationRelayClient.from_connection_string(connection_str)

    _ = identity_client.create_user()
    relay_client.get_relay_configuration()

    relay_configuration = relay_client.get_relay_configuration()

    for iceServer in relay_configuration.ice_servers:
        assert iceServer.username is not None
        assert iceServer.credential is not None
        assert iceServer.urls is not None
        for url in iceServer.urls:
            print('Url:' + url)

        credentials = {
            "username": iceServer.username,
            "credential": iceServer.credential
        }

    return func.HttpResponse(
            json.dumps(credentials),
            status_code=200
    )
