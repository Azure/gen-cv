# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import re
import time
import pytz
import uuid
import html
import json
import random
import requests
import datetime
import threading
import traceback
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from azure.identity import DefaultAzureCredential
from flask import Flask, Response, render_template, request

# Create the Flask app
app = Flask(__name__, template_folder='.')

# Environment variables
# Speech resource (required)
speech_region = os.environ.get('SPEECH_REGION')
speech_key = os.environ.get('SPEECH_KEY')
speech_private_endpoint = os.environ.get('SPEECH_PRIVATE_ENDPOINT') # (optional)
speech_resource_url = os.environ.get('SPEECH_RESOURCE_URL') # (optional, only used for private endpoint)
user_assigned_managed_identity_client_id = os.environ.get('USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID') # (optional, only used for private endpoint and user assigned managed identity)

# OpenAI resource (required for chat scenario)
azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
azure_openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY')
azure_openai_deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME')
azure_openai_system_prompt = os.environ.get('AZURE_OPENAI_SYSTEM_PROMPT')

# Cognitive search resource (optional, only required for 'on your data' scenario)
cognitive_search_endpoint = os.environ.get('COGNITIVE_SEARCH_ENDPOINT')
cognitive_search_api_key = os.environ.get('COGNITIVE_SEARCH_API_KEY')
cognitive_search_index_name = os.environ.get('COGNITIVE_SEARCH_INDEX_NAME')

# Customized ICE server (optional, only required for customized ICE server)
ice_server_url = os.environ.get('ICE_SERVER_URL') # The ICE URL, e.g. turn:x.x.x.x:3478
ice_server_url_remote = os.environ.get('ICE_SERVER_URL_REMOTE') # The ICE URL for remote side, e.g. turn:x.x.x.x:3478. This is only required when the ICE address for remote side is different from local side.
ice_server_username = os.environ.get('ICE_SERVER_USERNAME') # The ICE username
ice_server_password = os.environ.get('ICE_SERVER_PASSWORD') # The ICE password

# TTS Avatar Configuration
avatar_character = os.environ.get('AVATAR_CHARACTER')
avatar_style = os.environ.get('AVATAR_STYLE')
is_custom_avatar = os.environ.get('IS_CUSTOM_AVATAR')

background_color = "#1E1E1EFF"
background_image_url = None
transparent_background = "True"
video_crop = "False"

# Avatar Voice Configuration
tts_voice = os.environ.get('TTS_VOICE')
custom_voice_endpoint = os.environ.get('CUSTOM_VOICE_ENDPOINT') # optional
personal_voice_speaker_profile_id = os.environ.get('PERSONAL_VOICE_SPEAKER_PROFILE') # optional

# CSS Variables (Landscape)
webpage_background_landscape = os.environ.get('WEBPAGE_BACKGROUND_LANDSCAPE')
webpage_chat_fontcolor_landscape = os.environ.get('WEBPAGE_CHAT_FONTCOLOR_LANDSCAPE')
button_color_landscape = os.environ.get('BUTTON_COLOR_LANDSCAPE')
button_hover_color_landscape = os.environ.get('BUTTON_HOVER_LANDSCAPE')
button_icon_color_landscape = os.environ.get('BUTTON_ICON_COLOR_LANDSCAPE')

# CSS Variables (Portrait)
webpage_background_portrait = os.environ.get('WEBPAGE_BACKGROUND_PORTRAIT')
webpage_chat_fontcolor_portrait = os.environ.get('WEBPAGE_CHAT_FONTCOLOR_PORTRAIT')
button_color_portrait = os.environ.get('BUTTON_COLOR_PORTRAIT')
button_hover_color_portrait = os.environ.get('BUTTON_HOVER_PORTRAIT')
button_icon_color_portrait = os.environ.get('BUTTON_ICON_COLOR_PORTRAIT')

# Constant variables
sentence_level_punctuations = [ '.', '?', '!', ':', ';', '。', '?', '!', ':', ';' ] # Punctuations that indicate the end of a sentence
enable_quick_reply = False # Enable quick reply for certain chat models which take longer time to respond
quick_replies = [ 'Let me take a look.', 'Let me check.', 'One moment, please.' ] # Quick reply reponses
oyd_doc_regex = re.compile(r'\[doc(\d+)\]') # Regex to match the OYD (on-your-data) document reference
symbol_regex = re.compile(r'[#*_`]')

# Global variables
client_contexts = {} # Client contexts
speech_token = None # Speech token
ice_token = None # ICE token
azure_openai = AzureOpenAI(azure_endpoint=azure_openai_endpoint, api_version='2024-06-01', api_key=azure_openai_api_key)

# The default route (index.html)
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", client_id=initializeClient(), stt_locales="en-US", bg_img=webpage_background_landscape, chat_color=webpage_chat_fontcolor_landscape, button_color=button_color_landscape, button_hover_color=button_hover_color_landscape, button_icon_color=button_icon_color_landscape)

# The chat route (index.html)
@app.route("/chat", methods=["GET"])
def chatView():
    return render_template("index.html", client_id=initializeClient(), stt_locales="en-US", bg_img=webpage_background_landscape, chat_color=webpage_chat_fontcolor_landscape, button_color=button_color_landscape, button_hover_color=button_hover_color_landscape, button_icon_color=button_icon_color_landscape)

@app.route("/portrait", methods=["GET"])
def portraitView():
    return render_template("portrait.html", client_id=initializeClient(), stt_locales="en-US", bg_img=webpage_background_portrait, chat_color=webpage_chat_fontcolor_portrait, button_color=button_color_portrait, button_hover_color=button_hover_color_portrait, button_icon_color=button_icon_color_portrait)

# The API route to get the speech token
@app.route("/api/getSpeechToken", methods=["GET"])
def getSpeechToken() -> Response:
    global speech_token
    response = Response(speech_token, status=200)
    response.headers['SpeechRegion'] = speech_region
    if speech_private_endpoint:
        response.headers['SpeechPrivateEndpoint'] = speech_private_endpoint
    return response

# The API route to get the ICE token
@app.route("/api/getIceToken", methods=["GET"])
def getIceToken() -> Response:
    # Apply customized ICE server if provided
    if ice_server_url and ice_server_username and ice_server_password:
        custom_ice_token = json.dumps({
            'Urls': [ ice_server_url ],
            'Username': ice_server_username,
            'Password': ice_server_password
        })
        return Response(custom_ice_token, status=200)
    return Response(ice_token, status=200)

# The API route to connect the TTS avatar
@app.route("/api/connectAvatar", methods=["POST"])
def connectAvatar() -> Response:

    global client_contexts
    client_id = uuid.UUID(request.headers.get('ClientId'))
    client_context = client_contexts[client_id]

    # Override default values with client provided values
    client_context['azure_openai_deployment_name'] = azure_openai_deployment_name
    client_context['cognitive_search_index_name'] = cognitive_search_index_name
    client_context['tts_voice'] = tts_voice
    client_context['custom_voice_endpoint_id'] = custom_voice_endpoint
    client_context['personal_voice_speaker_profile_id'] = personal_voice_speaker_profile_id

    custom_voice_endpoint_id = client_context['custom_voice_endpoint_id']

    try:
        if speech_private_endpoint:
            speech_private_endpoint_wss = speech_private_endpoint.replace('https://', 'wss://')
            speech_config = speechsdk.SpeechConfig(subscription=speech_key, endpoint=f'{speech_private_endpoint_wss}/tts/cognitiveservices/websocket/v1?enableTalkingAvatar=true')
        else:
            speech_config = speechsdk.SpeechConfig(subscription=speech_key, endpoint=f'wss://{speech_region}.tts.speech.microsoft.com/cognitiveservices/websocket/v1?enableTalkingAvatar=true')

        if custom_voice_endpoint_id:
            speech_config.endpoint_id = custom_voice_endpoint_id

        client_context['speech_synthesizer'] = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        speech_synthesizer = client_context['speech_synthesizer']
        
        ice_token_obj = json.loads(ice_token)
        # Apply customized ICE server if provided
        if ice_server_url and ice_server_username and ice_server_password:
            ice_token_obj = {
                'Urls': [ ice_server_url_remote ] if ice_server_url_remote else [ ice_server_url ],
                'Username': ice_server_username,
                'Password': ice_server_password
            }

        local_sdp = request.data.decode('utf-8')        
        avatar_config = {
            'synthesis': {
                'video': {
                    'protocol': {
                        'name': "WebRTC",
                        'webrtcConfig': {
                            'clientDescription': local_sdp,
                            'iceServers': [{
                                'urls': [ ice_token_obj['Urls'][0] ],
                                'username': ice_token_obj['Username'],
                                'credential': ice_token_obj['Password']
                            }]
                        },
                    },
                    'format':{
                        'crop':{
                            'topLeft':{
                                'x': 600 if video_crop.lower() == 'true' else 0,
                                'y': 0
                            },
                            'bottomRight':{
                                'x': 1320 if video_crop.lower() == 'true' else 1920,
                                'y': 1080
                            }
                        },
                        'bitrate': 1000000
                    },
                    'talkingAvatar': {
                        'customized': is_custom_avatar.lower() == 'true',
                        'character': avatar_character,
                        'style': avatar_style,
                        # #00B140FF - chroma green, #009D57FF - television green, #0C8918FF - digital TV green
                        'background': {
                            'color': '#00B140FF' if transparent_background.lower() == 'true' else background_color,
                            'image': {
                                'url': background_image_url
                            }
                        }
                    }
                }
            }
        }
        
        connection = speechsdk.Connection.from_speech_synthesizer(speech_synthesizer)
        connection.set_message_property('speech.config', 'context', json.dumps(avatar_config))

        speech_sythesis_result = speech_synthesizer.speak_text_async('').get()
        print(f'Result id for avatar connection: {speech_sythesis_result.result_id}')
        if speech_sythesis_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_sythesis_result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
                raise Exception(cancellation_details.error_details)
        turn_start_message = speech_synthesizer.properties.get_property_by_name('SpeechSDKInternal-ExtraTurnStartMessage')
        remoteSdp = json.loads(turn_start_message)['webrtc']['connectionString']

        return Response(remoteSdp, status=200)

    except Exception as e:
        traceback.print_exc()
        return Response(f"Result ID: {speech_sythesis_result.result_id}. Error message: {e}", status=400)

# The API route to speak a given SSML
@app.route("/api/speak", methods=["POST"])
def speak() -> Response:
    client_id = uuid.UUID(request.headers.get('ClientId'))
    try:
        ssml = request.data.decode('utf-8')
        result_id = speakSsml(ssml, client_id, True)
        return Response(result_id, status=200)
    except Exception as e:
        return Response(f"Speak failed. Error message: {e}", status=400)

# The API route to stop avatar from speaking
@app.route("/api/stopSpeaking", methods=["POST"])
def stopSpeaking() -> Response:
    global client_contexts
    client_id = uuid.UUID(request.headers.get('ClientId'))
    is_speaking = client_contexts[client_id]['is_speaking']
    if is_speaking:
        stopSpeakingInternal(client_id)
    return Response('Speaking stopped.', status=200)

# The API route for chat
# It receives the user query and return the chat response.
# It returns response in stream, which yields the chat response in chunks.
@app.route("/api/chat", methods=["POST"])
def chat() -> Response:
    global client_contexts
    client_id = uuid.UUID(request.headers.get('ClientId'))
    client_context = client_contexts[client_id]
    chat_initiated = client_context['chat_initiated']
    if not chat_initiated:
        initializeChatContext(azure_openai_system_prompt, client_id)
        client_context['chat_initiated'] = True
    user_query = request.data.decode('utf-8')
    return Response(handleUserQuery(user_query, client_id), mimetype='text/plain', status=200)

# The API route to clear the chat history
@app.route("/api/chat/clearHistory", methods=["POST"])
def clearChatHistory() -> Response:
    global client_contexts
    client_id = uuid.UUID(request.headers.get('ClientId'))
    client_context = client_contexts[client_id]
    initializeChatContext(azure_openai_system_prompt, client_id)
    client_context['chat_initiated'] = True
    return Response('Chat history cleared.', status=200)

# The API route to disconnect the TTS avatar
@app.route("/api/disconnectAvatar", methods=["POST"])
def disconnectAvatar() -> Response:
    global client_contexts
    client_id = uuid.UUID(request.headers.get('ClientId'))
    client_context = client_contexts[client_id]
    speech_synthesizer = client_context['speech_synthesizer']
    try:
        connection = speechsdk.Connection.from_speech_synthesizer(speech_synthesizer)
        connection.close()
        return Response('Disconnected avatar', status=200)
    except:
        return Response(traceback.format_exc(), status=400)

# Initialize the client by creating a client id and an initial context
def initializeClient() -> uuid.UUID:
    client_id = uuid.uuid4()
    client_contexts[client_id] = {
        'speech_synthesizer': None, # Speech synthesizer for avatar
        'speech_token': None, # Speech token for client side authentication with speech service
        'ice_token': None, # ICE token for ICE/TURN/Relay server connection
        'chat_initiated': False, # Flag to indicate if the chat context is initiated
        'messages': [], # Chat messages (history)
        'is_speaking': False, # Flag to indicate if the avatar is speaking
        'spoken_text_queue': [], # Queue to store the spoken text
        'speaking_thread': None, # The thread to speak the spoken text queue
        'last_speak_time': None, # The last time the avatar spoke
        'data_sources': [] # Data sources for 'on your data' scenario
    }
    return client_id

# Refresh the ICE token which being called
def refreshIceToken() -> None:
    global ice_token
    if speech_private_endpoint:
        ice_token = requests.get(f'{speech_private_endpoint}/tts/cognitiveservices/avatar/relay/token/v1', headers={'Ocp-Apim-Subscription-Key': speech_key}).text
    else:
        ice_token = requests.get(f'https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/avatar/relay/token/v1', headers={'Ocp-Apim-Subscription-Key': speech_key}).text

# Refresh the speech token every 9 minutes
def refreshSpeechToken() -> None:
    global speech_token
    while True:
        # Refresh the speech token every 9 minutes
        if speech_private_endpoint:
            credential = DefaultAzureCredential(managed_identity_client_id=user_assigned_managed_identity_client_id)
            token = credential.get_token('https://cognitiveservices.azure.com/.default')
            speech_token = f'aad#{speech_resource_url}#{token.token}'
        else:
            speech_token = requests.post(f'https://{speech_region}.api.cognitive.microsoft.com/sts/v1.0/issueToken', headers={'Ocp-Apim-Subscription-Key': speech_key}).text
        time.sleep(60 * 9)

# Initialize the chat context, e.g. chat history (messages)
def initializeChatContext(system_prompt: str, client_id: uuid.UUID) -> None:
    global client_contexts
    client_context = client_contexts[client_id]
    messages = client_context['messages']
    data_sources = client_context['data_sources']

    # Initialize data sources for 'on your data' scenario
    data_sources.clear()
    if cognitive_search_endpoint and cognitive_search_api_key and cognitive_search_index_name:
        # On-your-data scenario
        data_source = {
            'type': 'azure_search',
            'parameters': {
                'endpoint': cognitive_search_endpoint,
                'index_name': cognitive_search_index_name,
                'authentication': {
                    'type': 'api_key',
                    'key': cognitive_search_api_key
                },
                'semantic_configuration': '',
                'query_type': 'simple',
                'fields_mapping': {
                    'content_fields_separator': '\n',
                    'content_fields': ['chunk'],
                    'filepath_field': None,
                    'title_field': 'title',
                    'url_field': None
                },
                'in_scope': True,
                'role_information': system_prompt
            }
        }
        data_sources.append(data_source)

    # Initialize messages
    messages.clear()
    if len(data_sources) == 0:
        system_message = {
            'role': 'system',
            'content': system_prompt
        }
        messages.append(system_message)


# Handle the user query and return the assistant reply. For chat scenario.
# The function is a generator, which yields the assistant reply in chunks.
def handleUserQuery(user_query: str, client_id: uuid.UUID):

    global client_contexts
    client_context = client_contexts[client_id]
    messages = client_context['messages']
    data_sources = client_context['data_sources']

    chat_message = {
        'role': 'user',
        'content': user_query
    }

    messages.append(chat_message)

    if enable_quick_reply:
        speakWithQueue(random.choice(quick_replies), 2000)

    assistant_reply = ''
    spoken_sentence = ''
    tool_content = ''

    aoai_start_time = datetime.datetime.now(pytz.UTC)
    response = azure_openai.chat.completions.create(
        model=azure_openai_deployment_name,
        messages=messages,
        extra_body={ 'data_sources' : data_sources } if len(data_sources) > 0 else None,
        max_tokens = 150,
        temperature = 0,
        stream=True)

    is_first_chunk = True
    is_first_sentence = True
    for chunk in response:
        if len(chunk.choices) > 0:
            response_token = chunk.choices[0].delta.content
            if response_token is not None:
                # Log response_token here if need debug
                if is_first_chunk:
                    first_token_latency_ms = round((datetime.datetime.now(pytz.UTC) - aoai_start_time).total_seconds() * 1000)
                    print(f"AOAI first token latency: {first_token_latency_ms}ms")
                    yield f"<FTL>{first_token_latency_ms}</FTL>"
                    is_first_chunk = False
                if oyd_doc_regex.search(response_token):
                    response_token = oyd_doc_regex.sub('', response_token).strip()
                if symbol_regex.search(response_token):
                    response_token = symbol_regex.sub('', response_token).strip()                
                yield response_token # yield response token to client as display text
                assistant_reply += response_token  # build up the assistant message
                if response_token == '\n' or response_token == '\n\n':
                    if is_first_sentence:
                        first_sentence_latency_ms = round((datetime.datetime.now(pytz.UTC) - aoai_start_time).total_seconds() * 1000)
                        print(f"AOAI first sentence latency: {first_sentence_latency_ms}ms")
                        yield f"<FSL>{first_sentence_latency_ms}</FSL>"
                        is_first_sentence = False
                    speakWithQueue(spoken_sentence.strip(), 0, client_id)
                    spoken_sentence = ''
                else:
                    response_token = response_token.replace('\n', '')
                    spoken_sentence += response_token  # build up the spoken sentence
                    if len(response_token) == 1 or len(response_token) == 2:
                        for punctuation in sentence_level_punctuations:
                            if response_token.startswith(punctuation):
                                if is_first_sentence:
                                    first_sentence_latency_ms = round((datetime.datetime.now(pytz.UTC) - aoai_start_time).total_seconds() * 1000)
                                    print(f"AOAI first sentence latency: {first_sentence_latency_ms}ms")
                                    yield f"<FSL>{first_sentence_latency_ms}</FSL>"
                                    is_first_sentence = False
                                speakWithQueue(spoken_sentence.strip(), 0, client_id)
                                spoken_sentence = ''
                                break

    if spoken_sentence != '':
        speakWithQueue(spoken_sentence.strip(), 0, client_id)
        spoken_sentence = ''

    if len(data_sources) > 0:
        tool_message = {
            'role': 'tool',
            'content': tool_content
        }
        messages.append(tool_message)

    assistant_message = {
        'role': 'assistant',
        'content': assistant_reply
    }
    messages.append(assistant_message)

# Speak the given text. If there is already a speaking in progress, add the text to the queue. For chat scenario.
def speakWithQueue(text: str, ending_silence_ms: int, client_id: uuid.UUID) -> None:
    global client_contexts
    client_context = client_contexts[client_id]
    spoken_text_queue = client_context['spoken_text_queue']
    is_speaking = client_context['is_speaking']
    spoken_text_queue.append(text)
    if not is_speaking:
        def speakThread():
            nonlocal client_context
            nonlocal spoken_text_queue
            nonlocal ending_silence_ms
            # tts_voice = client_context['tts_voice']
            # personal_voice_speaker_profile_id = client_context['personal_voice_speaker_profile_id']
            client_context['is_speaking'] = True
            while len(spoken_text_queue) > 0:
                text = spoken_text_queue.pop(0)
                speakText(text, tts_voice, personal_voice_speaker_profile_id, ending_silence_ms, client_id)
                client_context['last_speak_time'] = datetime.datetime.now(pytz.UTC)
            client_context['is_speaking'] = False
        client_context['speaking_thread'] = threading.Thread(target=speakThread)
        client_context['speaking_thread'].start()

# Speak the given text.
def speakText(text: str, voice: str, speaker_profile_id: str, ending_silence_ms: int, client_id: uuid.UUID) -> str:
    ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'>
                 <voice name='{voice}'>
                     <mstts:ttsembedding speakerProfileId='{speaker_profile_id}'>
                         <mstts:leadingsilence-exact value='0'/>
                         {html.escape(text)}
                     </mstts:ttsembedding>
                 </voice>
               </speak>"""
    if ending_silence_ms > 0:
        ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'>
                     <voice name='{voice}'>
                         <mstts:ttsembedding speakerProfileId='{speaker_profile_id}'>
                             <mstts:leadingsilence-exact value='0'/>
                             {html.escape(text)}
                             <break time='{ending_silence_ms}ms' />
                         </mstts:ttsembedding>
                     </voice>
                   </speak>"""
    return speakSsml(ssml, client_id, False)

# Speak the given ssml with speech sdk
def speakSsml(ssml: str, client_id: uuid.UUID, asynchronized: bool) -> str:
    global client_contexts
    speech_synthesizer = client_contexts[client_id]['speech_synthesizer']
    speech_sythesis_result = speech_synthesizer.start_speaking_ssml_async(ssml).get() if asynchronized else speech_synthesizer.speak_ssml_async(ssml).get()
    if speech_sythesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_sythesis_result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Result ID: {speech_sythesis_result.result_id}. Error details: {cancellation_details.error_details}")
            raise Exception(cancellation_details.error_details)
    return speech_sythesis_result.result_id

# Stop speaking internal function
def stopSpeakingInternal(client_id: uuid.UUID) -> None:
    global client_contexts
    client_context = client_contexts[client_id]
    speech_synthesizer = client_context['speech_synthesizer']
    spoken_text_queue = client_context['spoken_text_queue']
    spoken_text_queue.clear()
    try:
        connection = speechsdk.Connection.from_speech_synthesizer(speech_synthesizer)
        connection.send_message_async('synthesis.control', '{"action":"stop"}').get()
    except:
        print("Sending message through connection object is not yet supported by current Speech SDK.")

# Start the speech token refresh thread
speechTokenRefereshThread = threading.Thread(target=refreshSpeechToken)
speechTokenRefereshThread.daemon = True
speechTokenRefereshThread.start()

# Fetch ICE token at startup
refreshIceToken()