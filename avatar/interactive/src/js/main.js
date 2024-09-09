// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license.

const system_prompt = `
You are an AI assistant focused on delivering brief product details and assisting with the ordering process.
- Before calling a function, aim to answer product queries using the existing conversational context.
- If the product information isn't clear or available, consult get_product_information for accurate details. Never invent answers.  
- Address customer account or order-related queries with the appropriate functions.
- Before seeking account specifics (like account_id), scan previous parts of the conversation. Reuse information if available, avoiding repetitive queries.
- NEVER GUESS FUNCTION INPUTS! If a user's request is unclear, request further clarification.
- If not specified otherwise, the account_id of the current user is 1000
- Provide responses within 3 sentences for spoken output, emphasizing conciseness and accuracy.
- Formulate your response for spoken output. Do not output URLs. You can refer to the source like "XY National Park Website" BUT DO NOT use URLs
- IMPORTANT: Pay attention to the language the customer is using in their latest statement and ALWAYS respond in the same language!
`

var TTSVoice = "en-US-AvaMultilingualNeural" // Update this value if you want to use a different voices
const CogSvcRegion = "westeurope" // Fill your Azure cognitive services region here, e.g. westus2
var TalkingAvatarCharacter = "Meg"
var TalkingAvatarStyle = "formal"
const continuousRecording = false

supported_languages = ["en-US", "de-DE", "zh-CN", "nl-NL"] // The language detection engine supports a maximum of 4 languages

const speechSynthesisConfig = SpeechSDK.SpeechConfig.fromEndpoint(new URL("wss://{region}.tts.speech.microsoft.com/cognitiveservices/websocket/v1?enableTalkingAvatar=true".replace("{region}", CogSvcRegion)))

// Global objects
var speechSynthesizer
var avatarSynthesizer
var peerConnection
var previousAnimationFrameTimestamp = 0
var messages = [{ "role": "system", "content": system_prompt }];
var sentenceLevelPunctuations = ['.', '?', '!', ':', ';', '。', '？', '！', '：', '；']
var isSpeaking = false
var spokenTextQueue = []
var lastSpeakTime
let token

// Setup WebRTC
function setupWebRTC() {
  // Create WebRTC peer connection
  fetch("/api/get-ice-server-token", {
    method: "POST"
  })
    .then(async res => {
      const reponseJson = await res.json()
      peerConnection = new RTCPeerConnection({
        iceServers: [{
          urls: reponseJson["Urls"],
          username: reponseJson["Username"],
          credential: reponseJson["Password"]
        }]
      })

      // Fetch WebRTC video stream and mount it to an HTML video element
      peerConnection.ontrack = function (event) {
        console.log('peerconnection.ontrack', event)
        // Clean up existing video element if there is any
        remoteVideoDiv = document.getElementById('remoteVideo')
        for (var i = 0; i < remoteVideoDiv.childNodes.length; i++) {
          if (remoteVideoDiv.childNodes[i].localName === event.track.kind) {
            remoteVideoDiv.removeChild(remoteVideoDiv.childNodes[i])
          }
        }

        const videoElement = document.createElement(event.track.kind)
        videoElement.id = event.track.kind
        videoElement.srcObject = event.streams[0]
        videoElement.autoplay = true
        videoElement.controls = false
        document.getElementById('remoteVideo').appendChild(videoElement)

        canvas = document.getElementById('canvas')
        remoteVideoDiv.hidden = true
        canvas.hidden = false

        videoElement.addEventListener('play', () => {
          remoteVideoDiv.style.width = videoElement.videoWidth / 2 + 'px'
          window.requestAnimationFrame(makeBackgroundTransparent)
        })
      }

      // Make necessary update to the web page when the connection state changes
      peerConnection.oniceconnectionstatechange = e => {
        console.log("WebRTC status: " + peerConnection.iceConnectionState)

        if (peerConnection.iceConnectionState === 'connected') {
          document.getElementById('loginOverlay').classList.add("hidden");
        }

        if (peerConnection.iceConnectionState === 'disconnected') {
        }
      }

      // Offer to receive 1 audio, and 1 video track
      peerConnection.addTransceiver('video', { direction: 'sendrecv' })
      peerConnection.addTransceiver('audio', { direction: 'sendrecv' })

      // start avatar, establish WebRTC connection
      avatarSynthesizer.startAvatarAsync(peerConnection).then((r) => {
        if (r.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
          console.log("[" + (new Date()).toISOString() + "] Avatar started. Result ID: " + r.resultId)
          greeting()
        } else {
          console.log("[" + (new Date()).toISOString() + "] Unable to start avatar. Result ID: " + r.resultId)
          if (r.reason === SpeechSDK.ResultReason.Canceled) {
            let cancellationDetails = SpeechSDK.CancellationDetails.fromResult(r)
            if (cancellationDetails.reason === SpeechSDK.CancellationReason.Error) {
              console.log(cancellationDetails.errorDetails)
            };

            console.log("Unable to start avatar: " + cancellationDetails.errorDetails);
          }
        }
      }).catch(
        (error) => {
          console.log("[" + (new Date()).toISOString() + "] Avatar failed to start. Error: " + error)
          document.getElementById('startSession').disabled = false
          document.getElementById('configuration').hidden = false
        }
      )

    })
}

function handleUserQuery(userQuery, userQueryHTML) {
  let contentMessage = userQuery
  console.log('handleUserQuery', contentMessage)

  let chatMessage = {
    role: 'user',
    content: contentMessage
  }

  messages.push(chatMessage)
  addToConversationHistory(contentMessage, 'dark')
  if (isSpeaking) {
    stopSpeaking()
  }

  let body = JSON.stringify({
    messages: messages
  })

  let assistantReply = ''
  let toolContent = ''
  let spokenSentence = ''
  let displaySentence = ''

  fetch("/api/get-oai-response", {
    method: "POST",
    body: body
  })
    .then(response => {
      if (!response.ok) {
        throw new Error(`Chat API response status: ${response.status} ${response.statusText}`)
      }

      const reader = response.body.getReader()

      // Function to recursively read chunks from the stream
      function read(previousChunkString = '') {
        return reader.read().then(({ value, done }) => {
          // Check if there is still data to read
          if (done) {
            // Stream complete
            return
          }

          // Process the chunk of data (value)
          let chunkString = new TextDecoder().decode(value, { stream: true })
          if (previousChunkString !== '') {
            // Concatenate the previous chunk string in case it is incomplete
            chunkString = previousChunkString + chunkString
          }

          new TextDecoder().decode(value, { stream: true, json: true})

          try {
            responseToken = chunkString
            console.log('responseToken', responseToken)
            
            if (responseToken !== undefined && responseToken !== null) {
              try {
                const isObject = (x) => typeof x === 'object' && !Array.isArray(x) && x !== null
                console.log(responseToken, typeof responseToken)
                product = JSON.parse(responseToken)
                console.log(product, isObject(product), typeof product)
                if (isObject(product)) {
                  addProductToChatHistory(product)
                  console.log(product)
                  responseToken = ''
                }
              } catch (error) {
                console.log('Error parsing product:', error)
              }
              assistantReply += responseToken // build up the assistant message
              displaySentence += responseToken // build up the display sentence

              if (responseToken === '\n' || responseToken === '\n\n') {
                speak(spokenSentence.trim())
                spokenSentence = ''
              } else {
                responseToken = responseToken.replace(/\n/g, '')
                responseToken = responseToken.replace(/[*\uD83C-\uDBFF\uDC00-\uDFFF]+/g, '');
                spokenSentence += responseToken // build up the spoken sentence

                if (responseToken.length === 1 || responseToken.length === 2) {
                  for (let i = 0; i < sentenceLevelPunctuations.length; ++i) {
                    let sentenceLevelPunctuation = sentenceLevelPunctuations[i]
                    if (responseToken.startsWith(sentenceLevelPunctuation)) {
                      speak(spokenSentence.trim())
                      spokenSentence = ''
                      break
                    }
                  }
                }
              }
            }
          } catch (error) {
            console.log(`Error occurred while parsing the response: ${error}`)
            console.log(chunkString)
          }
          // })

          if (displaySentence !== '') {
            addToConversationHistory(displaySentence, 'light');
          }
          displaySentence = ''
          return read()
        })
      }

      // Start reading the stream
      return read()
    })
    .then(() => {
      if (spokenSentence !== '') {
        speak(spokenSentence.trim())
        spokenSentence = ''
      }
      let assistantMessage = {
        role: 'assistant',
        content: assistantReply
      }

      messages.push(assistantMessage)
    })
}

// Speak the given text
function speak(text, endingSilenceMs = 0) {
  if (isSpeaking) {
    spokenTextQueue.push(text)
    return
  }

  speakNext(text, endingSilenceMs)
}

function speakNext(text, endingSilenceMs = 0) {
  let ssml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'><voice name='${TTSVoice}'><mstts:leadingsilence-exact value='0'/>${text}</voice></speak>`
  if (endingSilenceMs > 0) {
    ssml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'><voice name='${TTSVoice}'><mstts:leadingsilence-exact value='0'/>${text}<break time='${endingSilenceMs}ms' /></voice></speak>`
  }

  lastSpeakTime = new Date()
  isSpeaking = true
  avatarSynthesizer.speakSsmlAsync(ssml).then(
    (result) => {
      if (result.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
        console.log(`Speech synthesized to speaker for text [ ${text} ]. Result ID: ${result.resultId}`)

        lastSpeakTime = new Date()
      } else {
        console.log(`Error occurred while speaking the SSML. Result ID: ${result.resultId}`)
      }

      if (spokenTextQueue.length > 0) {
        speakNext(spokenTextQueue.shift())
      } else {
        isSpeaking = false
      }
    }).catch(
      (error) => {
        console.log(`Error occurred while speaking the SSML: [ ${error} ]`)

        if (spokenTextQueue.length > 0) {
          speakNext(spokenTextQueue.shift())
        } else {
          isSpeaking = false
        }
      }
    )
}

function stopSpeaking() {
  spokenTextQueue = []
  avatarSynthesizer.stopSpeakingAsync().then(
    () => {
      isSpeaking = false
      console.log("[" + (new Date()).toISOString() + "] Stop speaking request sent.")
    }
  ).catch(
    (error) => {
      console.log("Error occurred while stopping speaking: " + error)
    }
  )
}


// Connect to TTS Avatar API
function connectToAvatarService() {
  // Construct TTS Avatar service request
  let videoCropTopLeftX = 600
  let videoCropBottomRightX = 1320
  let backgroundColor = '#00FF00FF'

  const videoFormat = new SpeechSDK.AvatarVideoFormat()
  videoFormat.setCropRange(new SpeechSDK.Coordinate(videoCropTopLeftX, 0), new SpeechSDK.Coordinate(videoCropBottomRightX, 1080));

  TalkingAvatarCharacter = document.getElementById("avatar-name").value
  switch(TalkingAvatarCharacter) {
    case "Lisa":
      TalkingAvatarStyle = "casual-sitting"
      break
    case "Meg":
      TalkingAvatarStyle = "casual"
      break
    case "Mark":
      TalkingAvatarStyle = "formal"
      break
     
  }

  const avatarConfig = new SpeechSDK.AvatarConfig(TalkingAvatarCharacter, TalkingAvatarStyle, videoFormat)
  avatarConfig.backgroundColor = backgroundColor

  avatarSynthesizer = new SpeechSDK.AvatarSynthesizer(speechSynthesisConfig, avatarConfig)
  avatarSynthesizer.avatarEventReceived = function (s, e) {
    var offsetMessage = ", offset from session start: " + e.offset / 10000 + "ms."
    if (e.offset === 0) {
      offsetMessage = ""
    }
    console.log("Event received: " + e.description + offsetMessage)
  }

}

window.startSession = () => {
  var iconElement = document.createElement("i");
  iconElement.className = "fa fa-spinner fa-spin";
  iconElement.id = "loadingIcon"
  var parentElement = document.getElementById("playVideo");
  parentElement.prepend(iconElement);

  TTSVoice = document.getElementById("avatar-voice").value

  speechSynthesisConfig.speechSynthesisVoiceName = TTSVoice
  document.getElementById('playVideo').className = "round-button-hide"

  fetch("/api/get-speech-token", {
    method: "GET"
  })
    .then(async res => {
      const responseJson = await res.json()
      speechSynthesisConfig.authorizationToken = responseJson.token;
      token = responseJson.token
    })
    .then(() => {
      speechSynthesizer = new SpeechSDK.SpeechSynthesizer(speechSynthesisConfig, null)
      connectToAvatarService()
      setupWebRTC()
    })
}

async function greeting() {
  text = `Hi, my name is ${TalkingAvatarCharacter}. How can I help you?`;
  addToConversationHistory(text, "light")

  var spokenText = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'><voice name='${TTSVoice}'><mstts:leadingsilence-exact value='0'/>${text}</voice></speak>`

  console.log('spokenText', spokenText)
  avatarSynthesizer.speakSsmlAsync(spokenText, (result) => {
    if (result.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
      console.log("Speech synthesized to speaker for text [ " + spokenText + " ]. Result ID: " + result.resultId)
    } else {
      console.log("Unable to speak text. Result ID: " + result.resultId)
      if (result.reason === SpeechSDK.ResultReason.Canceled) {
        let cancellationDetails = SpeechSDK.CancellationDetails.fromResult(result)
        console.log(cancellationDetails.reason)
        if (cancellationDetails.reason === SpeechSDK.CancellationReason.Error) {
          console.log(cancellationDetails.errorDetails)
        }
      }
    }
  })
}

window.stopSession = () => {
  speechSynthesizer.close()
}

window.startRecording = () => {
  const speechConfig = SpeechSDK.SpeechConfig.fromAuthorizationToken(token, 'westeurope');
  speechConfig.authorizationToken = token;
  speechConfig.SpeechServiceConnection_LanguageIdMode = "Continuous";
  var autoDetectSourceLanguageConfig = SpeechSDK.AutoDetectSourceLanguageConfig.fromLanguages(supported_languages);

  document.getElementById('buttonIcon').className = "fas fa-stop"
  document.getElementById('startRecording').disabled = true

  recognizer = SpeechSDK.SpeechRecognizer.FromConfig(speechConfig, autoDetectSourceLanguageConfig);

  recognizer.recognized = function (s, e) {
    if (e.result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
      let userQuery = e.result.text.trim()
      if (userQuery === '') {
        return
      }
      console.log('Recognized:', e.result.text);
      if (!continuousRecording) {
        window.stopRecording();
      }

      handleUserQuery(e.result.text, "", "")
    }
  };

  recognizer.startContinuousRecognitionAsync();

  console.log('Recording started.');
}

window.stopRecording = () => {
  if (recognizer) {
    recognizer.stopContinuousRecognitionAsync(
      function () {
        recognizer.close();
        recognizer = undefined;
        document.getElementById('buttonIcon').className = "fas fa-microphone"
        document.getElementById('startRecording').disabled = false
        console.log('Recording stopped.');
      },
      function (err) {
        console.error('Error stopping recording:', err);
      }
    );
  }
}

window.submitText = () => {
  document.getElementById('spokenText').textContent = document.getElementById('textinput').currentValue
  document.getElementById('textinput').currentValue = ""
  window.speak(document.getElementById('textinput').currentValue);
}

function addToConversationHistory(item, historytype) {
  const list = document.getElementById('chathistory');
  if (list.children.length !== 0) {
    const lastItem = list.lastChild;
    if (lastItem.classList.contains(`message--${historytype}`)) {
      lastItem.textContent += `${item}`;
      return;
    }
  }
  const newItem = document.createElement('li');
  newItem.classList.add('message');
  newItem.classList.add(`message--${historytype}`);
  newItem.textContent = item;
  list.appendChild(newItem);
}

function addProductToChatHistory(product) {
  const list = document.getElementById('chathistory');
  const listItem = document.createElement('li');
  listItem.classList.add('product');
  listItem.innerHTML = `
    <fluent-card class="product-card">
      <div class="product-card__header">
        <img src="${product.image_url}" alt="tent" width="100%">
      </div>
      <div class="product-card__content">
        <div><span class="product-card__price">$${product.special_offer}</span> <span class="product-card__old-price">$${product.original_price}</span></div>
        <div>${product.tagline}</div>
      </div>
    </fluent-card>
  `;
  list.appendChild(listItem);
}

// Make video background transparent by matting
function makeBackgroundTransparent(timestamp) {
  // Throttle the frame rate to 30 FPS to reduce CPU usage
  if (timestamp - previousAnimationFrameTimestamp > 30) {
    video = document.getElementById('video')
    tmpCanvas = document.getElementById('tmpCanvas')
    tmpCanvasContext = tmpCanvas.getContext('2d', { willReadFrequently: true })
    tmpCanvasContext.drawImage(video, 0, 0, video.videoWidth, video.videoHeight)
    if (video.videoWidth > 0) {
      let frame = tmpCanvasContext.getImageData(0, 0, video.videoWidth, video.videoHeight)
      for (let i = 0; i < frame.data.length / 4; i++) {
        let r = frame.data[i * 4 + 0]
        let g = frame.data[i * 4 + 1]
        let b = frame.data[i * 4 + 2]

        if (g - 150 > r + b) {
          // Set alpha to 0 for pixels that are close to green
          frame.data[i * 4 + 3] = 0
        } else if (g + g > r + b) {
          // Reduce green part of the green pixels to avoid green edge issue
          adjustment = (g - (r + b) / 2) / 3
          r += adjustment
          g -= adjustment * 2
          b += adjustment
          frame.data[i * 4 + 0] = r
          frame.data[i * 4 + 1] = g
          frame.data[i * 4 + 2] = b
          // Reduce alpha part for green pixels to make the edge smoother
          a = Math.max(0, 255 - adjustment * 4)
          frame.data[i * 4 + 3] = a
        }
      }

      canvas = document.getElementById('canvas')
      canvasContext = canvas.getContext('2d')
      canvasContext.putImageData(frame, 0, 0);
    }

    previousAnimationFrameTimestamp = timestamp
  }

  window.requestAnimationFrame(makeBackgroundTransparent)
}
