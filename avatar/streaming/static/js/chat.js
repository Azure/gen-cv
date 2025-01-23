// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license.

// Global objects
var clientId
var speechRecognizer
var peerConnection
var isSpeaking = false
var sessionActive = false
var recognitionStartedTime
var chatResponseReceivedTime
var lastSpeakTime
var isFirstRecognizingEvent = true
var firstTokenLatencyRegex = new RegExp(/<FTL>(\d+)<\/FTL>/)
var firstSentenceLatencyRegex = new RegExp(/<FSL>(\d+)<\/FSL>/)
var previousAnimationFrameTimestamp = 0;

// Connect to avatar service
function connectAvatar() {
    document.getElementById('startSession').disabled = true

    fetch('/api/getIceToken', {
        method: 'GET',
    })
    .then(response => {
        if (response.ok) {
            response.json().then(data => {
                const iceServerUrl = data.Urls[0]
                const iceServerUsername = data.Username
                const iceServerCredential = data.Password
                setupWebRTC(iceServerUrl, iceServerUsername, iceServerCredential)
            })
        } else {
            throw new Error(`Failed fetching ICE token: ${response.status} ${response.statusText}`)
        }
    })

    document.getElementById('configuration').hidden = true
}

// Create speech recognizer
function createSpeechRecognizer() {
    fetch('/api/getSpeechToken', {
        method: 'GET',
    })
    .then(response => {
        if (response.ok) {
            const speechRegion = response.headers.get('SpeechRegion')
            const speechPrivateEndpoint = response.headers.get('SpeechPrivateEndpoint')
            response.text().then(text => {
                const speechToken = text
                const speechRecognitionConfig = speechPrivateEndpoint ?
                    SpeechSDK.SpeechConfig.fromEndpoint(new URL(`wss://${speechPrivateEndpoint.replace('https://', '')}/stt/speech/universal/v2`), '') :
                    SpeechSDK.SpeechConfig.fromEndpoint(new URL(`wss://${speechRegion}.stt.speech.microsoft.com/speech/universal/v2`), '')
                speechRecognitionConfig.authorizationToken = speechToken
                speechRecognitionConfig.setProperty(SpeechSDK.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous")
                speechRecognitionConfig.setProperty("SpeechContext-PhraseDetection.TrailingSilenceTimeout", "3000")
                speechRecognitionConfig.setProperty("SpeechContext-PhraseDetection.InitialSilenceTimeout", "10000")
                speechRecognitionConfig.setProperty("SpeechContext-PhraseDetection.Dictation.Segmentation.Mode", "Custom")
                speechRecognitionConfig.setProperty("SpeechContext-PhraseDetection.Dictation.Segmentation.SegmentationSilenceTimeoutMs", "200")
                var sttLocales = document.getElementById('sttLocales').value.split(',')
                var autoDetectSourceLanguageConfig = SpeechSDK.AutoDetectSourceLanguageConfig.fromLanguages(sttLocales)
                speechRecognizer = SpeechSDK.SpeechRecognizer.FromConfig(speechRecognitionConfig, autoDetectSourceLanguageConfig, SpeechSDK.AudioConfig.fromDefaultMicrophoneInput())
            })
        } else {
            throw new Error(`Failed fetching speech token: ${response.status} ${response.statusText}`)
        }
    })
}

// Disconnect from avatar service
function disconnectAvatar(closeSpeechRecognizer = false) {
    fetch('/api/disconnectAvatar', {
        method: 'POST',
        headers: {
            'ClientId': clientId
        },
        body: ''
    })

    if (speechRecognizer !== undefined) {
        speechRecognizer.stopContinuousRecognitionAsync()
        if (closeSpeechRecognizer) {
            speechRecognizer.close()
        }
    }

    sessionActive = false
}

// Setup WebRTC
function setupWebRTC(iceServerUrl, iceServerUsername, iceServerCredential) {
    // Create WebRTC peer connection
    peerConnection = new RTCPeerConnection({
        iceServers: [{
            urls: [ iceServerUrl ],
            username: iceServerUsername,
            credential: iceServerCredential
        }],
        iceTransportPolicy: 'relay'
    })

    // Fetch WebRTC video stream and mount it to an HTML video element
    peerConnection.ontrack = function (event) {
        if (event.track.kind === 'audio') {
            let audioElement = document.createElement('audio')
            audioElement.id = 'audioPlayer'
            audioElement.srcObject = event.streams[0]
            audioElement.autoplay = true

            audioElement.onplaying = () => {
                console.log(`WebRTC ${event.track.kind} channel connected.`)
            }

            // Clean up existing audio element if there is any
            remoteVideoDiv = document.getElementById('remoteVideo')
            for (var i = 0; i < remoteVideoDiv.childNodes.length; i++) {
                if (remoteVideoDiv.childNodes[i].localName === event.track.kind) {
                    remoteVideoDiv.removeChild(remoteVideoDiv.childNodes[i])
                }
            }

            // Append the new audio element
            document.getElementById('remoteVideo').appendChild(audioElement)
        }

        if (event.track.kind === 'video') {
            let videoElement = document.createElement('video')
            videoElement.id = 'videoPlayer'
            videoElement.srcObject = event.streams[0]
            videoElement.autoplay = true
            videoElement.playsInline = true

            videoElement.onplaying = () => {
                // Clean up existing video element if there is any
                remoteVideoDiv = document.getElementById('remoteVideo')
                canvas = document.getElementById('canvas')
                remoteVideoDiv.style.width = '0.1px'
                canvas.hidden = false

                for (var i = 0; i < remoteVideoDiv.childNodes.length; i++) {
                    if (remoteVideoDiv.childNodes[i].localName === event.track.kind) {
                        remoteVideoDiv.removeChild(remoteVideoDiv.childNodes[i])
                    }
                }

                window.requestAnimationFrame(makeBackgroundTransparent)

                // Append the new video element
                document.getElementById('remoteVideo').appendChild(videoElement)

                console.log(`WebRTC ${event.track.kind} channel connected.`)
                document.getElementById('microphone').disabled = false
                document.getElementById('stopSession').disabled = false
                document.getElementById('chatHistory').hidden = false
                document.getElementById('latencyLog').hidden = false
                document.getElementById('showTypeMessage').disabled = false

                if (document.getElementById('useLocalVideoForIdle').checked) {
                    document.getElementById('localVideo').hidden = true
                    if (lastSpeakTime === undefined) {
                        lastSpeakTime = new Date()
                    }
                }

                setTimeout(() => { sessionActive = true }, 5000) // Set session active after 5 seconds
            }
        }
    }

    // Listen to data channel, to get the event from the server
    peerConnection.addEventListener("datachannel", event => {
        const dataChannel = event.channel
        dataChannel.onmessage = e => {
            console.log("[" + (new Date()).toISOString() + "] WebRTC event received: " + e.data)

            if (e.data.includes("EVENT_TYPE_SWITCH_TO_SPEAKING")) {
                if (chatResponseReceivedTime !== undefined) {
                    let speakStartTime = new Date()
                    let ttsLatency = speakStartTime - chatResponseReceivedTime
                    console.log(`TTS latency: ${ttsLatency} ms`)
                    let latencyLogTextArea = document.getElementById('latencyLog')
                    latencyLogTextArea.innerHTML += `TTS latency: ${ttsLatency} ms\n\n`
                    latencyLogTextArea.scrollTop = latencyLogTextArea.scrollHeight
                    chatResponseReceivedTime = undefined
                }

                isSpeaking = true
                document.getElementById('stopSpeaking').disabled = false
            } else if (e.data.includes("EVENT_TYPE_SWITCH_TO_IDLE")) {
                isSpeaking = false
                lastSpeakTime = new Date()
                document.getElementById('stopSpeaking').disabled = true
            }
        }
    })

    // This is a workaround to make sure the data channel listening is working by creating a data channel from the client side
    c = peerConnection.createDataChannel("eventChannel")

    // Make necessary update to the web page when the connection state changes
    peerConnection.oniceconnectionstatechange = e => {
        console.log("WebRTC status: " + peerConnection.iceConnectionState)
        if (peerConnection.iceConnectionState === 'disconnected') {
            if (document.getElementById('useLocalVideoForIdle').checked) {
                document.getElementById('localVideo').hidden = false
                document.getElementById('remoteVideo').style.width = '0.1px'
            }
        }
    }

    // Offer to receive 1 audio, and 1 video track
    peerConnection.addTransceiver('video', { direction: 'sendrecv' })
    peerConnection.addTransceiver('audio', { direction: 'sendrecv' })

    // Connect to avatar service when ICE candidates gathering is done
    iceGatheringDone = false

    peerConnection.onicecandidate = e => {
        if (!e.candidate && !iceGatheringDone) {
            iceGatheringDone = true
            connectToAvatarService(peerConnection)
        }
    }

    peerConnection.createOffer().then(sdp => {
        peerConnection.setLocalDescription(sdp).then(() => { setTimeout(() => {
            if (!iceGatheringDone) {
                iceGatheringDone = true
                connectToAvatarService(peerConnection)
            }
        }, 2000) })
    })
}

// Connect to TTS Avatar Service
function connectToAvatarService(peerConnection) {
    let localSdp = btoa(JSON.stringify(peerConnection.localDescription))
    let headers = {
        'ClientId': clientId
    }

    fetch('/api/connectAvatar', {
        method: 'POST',
        headers: headers,
        body: localSdp
    })
    .then(response => {
        if (response.ok) {
            response.text().then(text => {
                const remoteSdp = text
                peerConnection.setRemoteDescription(new RTCSessionDescription(JSON.parse(atob(remoteSdp))))
                .then(() => {
                    sendGreetingMessage();
                });
            })
        } else {
            document.getElementById('startSession').disabled = false;
            document.getElementById('configuration').hidden = false;
            throw new Error(`Failed connecting to the Avatar service: ${response.status} ${response.statusText}`)
        }
    })
}

// Function to send a greeting message
function sendGreetingMessage() {
    const greetingText = `Hi, my name is Lisa. How can I help you today?`;
    let ssml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'><voice name='en-US-NovaTurboMultilingualNeural'><mstts:leadingsilence-exact value='0'/>${greetingText}</voice></speak>`;
    
    fetch('/api/speak', {
        method: 'POST',
        headers: {
            'ClientId': clientId,
            'Content-Type': 'application/ssml+xml'
        },
        body: ssml
    })
    .then(response => {
        if (response.ok) {
            console.log('Greeting message sent successfully.');
        } else {
            console.error('Failed to send greeting message.');
        }
    });
}

// Handle user query. Send user query to the chat API and display the response.
function handleUserQuery(userQuery) {
    let chatRequestSentTime = new Date()
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'ClientId': clientId,
            'Content-Type': 'text/plain'
        },
        body: userQuery
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Chat API response status: ${response.status} ${response.statusText}`)
        }

        let chatHistoryTextArea = document.getElementById('chatHistory')
        chatHistoryTextArea.innerHTML += 'Assistant: '

        const reader = response.body.getReader()

        // Function to recursively read chunks from the stream
        function read() {
            return reader.read().then(({ value, done }) => {
                // Check if there is still data to read
                if (done) {
                    // Stream complete
                    return
                }

                // Process the chunk of data (value)
                let chunkString = new TextDecoder().decode(value, { stream: true })

                if (firstTokenLatencyRegex.test(chunkString)) {
                    let aoaiFirstTokenLatency = parseInt(firstTokenLatencyRegex.exec(chunkString)[0].replace('<FTL>', '').replace('</FTL>', ''))
                    // console.log(`AOAI first token latency: ${aoaiFirstTokenLatency} ms`)
                    chunkString = chunkString.replace(firstTokenLatencyRegex, '')
                    if (chunkString === '') {
                        return read()
                    }
                }

                if (firstSentenceLatencyRegex.test(chunkString)) {
                    let aoaiFirstSentenceLatency = parseInt(firstSentenceLatencyRegex.exec(chunkString)[0].replace('<FSL>', '').replace('</FSL>', ''))
                    chatResponseReceivedTime = new Date()
                    let chatLatency = chatResponseReceivedTime - chatRequestSentTime
                    let appServiceLatency = chatLatency - aoaiFirstSentenceLatency
                    console.log(`App service latency: ${appServiceLatency} ms`)
                    console.log(`AOAI latency: ${aoaiFirstSentenceLatency} ms`)
                    let latencyLogTextArea = document.getElementById('latencyLog')
                    latencyLogTextArea.innerHTML += `App service latency: ${appServiceLatency} ms\n`
                    latencyLogTextArea.innerHTML += `AOAI latency: ${aoaiFirstSentenceLatency} ms\n`
                    latencyLogTextArea.scrollTop = latencyLogTextArea.scrollHeight
                    chunkString = chunkString.replace(firstSentenceLatencyRegex, '')
                    if (chunkString === '') {
                        return read()
                    }
                }

                chatHistoryTextArea.innerHTML += `${chunkString}`
                chatHistoryTextArea.scrollTop = chatHistoryTextArea.scrollHeight

                // Continue reading the next chunk
                return read()
            })
        }

        // Start reading the stream
        return read()
    })
}

// Handle local video. If the user is not speaking for 15 seconds, switch to local video.
function handleLocalVideo() {
    if (lastSpeakTime === undefined) {
        return
    }

    let currentTime = new Date()
    if (currentTime - lastSpeakTime > 15000) {
        if (document.getElementById('useLocalVideoForIdle').checked && sessionActive && !isSpeaking) {
            disconnectAvatar()
            document.getElementById('localVideo').hidden = false
            document.getElementById('remoteVideo').style.width = '0.1px'
            sessionActive = false
        }
    }
}

// Check whether the avatar video stream is hung
function checkHung() {
    // Check whether the avatar video stream is hung, by checking whether the video time is advancing
    let videoElement = document.getElementById('videoPlayer')
    if (videoElement !== null && videoElement !== undefined && sessionActive) {
        let videoTime = videoElement.currentTime
        setTimeout(() => {
            // Check whether the video time is advancing
            if (videoElement.currentTime === videoTime) {
                // Check whether the session is active to avoid duplicatedly triggering reconnect
                if (sessionActive) {
                    sessionActive = false
                    if (document.getElementById('autoReconnectAvatar').checked) {
                        console.log(`[${(new Date()).toISOString()}] The video stream got disconnected, need reconnect.`)
                        connectAvatar()
                        createSpeechRecognizer()
                    }
                }
            }
        }, 2000)
    }
}

function checkAndExecute() {
    var checkbox = document.getElementById('showTypeMessage');
    if (checkbox.checked) {
      window.updateTypeMessageBox();
    }
  }

function makeBackgroundTransparent(timestamp) {
    if (!previousAnimationFrameTimestamp || timestamp - previousAnimationFrameTimestamp > 33) {
        const video = document.getElementById('videoPlayer');
        const canvas = document.getElementById('canvas');
        if (video && canvas && video.videoWidth > 0 && video.videoHeight > 0) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');

            // Clear the canvas
            context.clearRect(0, 0, canvas.width, canvas.height);

            // Draw the video frame onto the canvas
            context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
            let frame = context.getImageData(0, 0, video.videoWidth, video.videoHeight);

            // Process each pixel
            for (let i = 0; i < frame.data.length; i += 4) {
                let r = frame.data[i];
                let g = frame.data[i + 1];
                let b = frame.data[i + 2];
                let alpha = frame.data[i + 3];

                // Strict green detection for transparency
                if (g > 100 && g > r * 1.6 && g > b * 1.6) {
                    frame.data[i + 3] = 0; // Fully transparent for dominant green
                } 
                // Soften edge pixels
                else if (g > 80 && g > r * 1.4 && g > b * 1.4) {
                    frame.data[i + 3] = alpha * 0.1; // Partially transparent for soft edges
                }

                // Green spill reduction
                if (alpha > 0 && g > r * 1.2 && g > b * 1.2) {
                    let adjustment = (g - Math.max(r, b)) / 2;
                    frame.data[i] = Math.min(255, r + adjustment * 1.0); // Boost red
                    frame.data[i + 1] = Math.max(0, g - adjustment * 2.0); // Reduce green
                    frame.data[i + 2] = Math.min(255, b + adjustment * 1.0); // Boost blue
                }
            }

            context.putImageData(frame, 0, 0);

            // Apply edge-specific smoothing
            smoothEdges(context, canvas.width, canvas.height);
        }
        previousAnimationFrameTimestamp = timestamp;
    }
    window.requestAnimationFrame(makeBackgroundTransparent);
}

function smoothEdges(context, width, height) {
    let frame = context.getImageData(0, 0, width, height);
    let data = frame.data;
    
    for (let i = 0; i < data.length; i += 4) {
        let alpha = data[i + 3];

        // Only apply smoothing to semi-transparent pixels
        if (alpha > 0 && alpha < 255) {
            let surroundingAlpha = getSurroundingAlphaAverage(data, i, width);
            // Smooth the alpha value by blending it with the surrounding pixels' alpha
            data[i + 3] = (alpha + surroundingAlpha) / 2;
        }
    }

    context.putImageData(frame, 0, 0);
}

function getSurroundingAlphaAverage(data, index, width) {
    let totalAlpha = 0;
    let count = 0;

    // Check surrounding pixels in a 3x3 grid
    for (let y = -1; y <= 1; y++) {
        for (let x = -1; x <= 1; x++) {
            let neighborIndex = index + (y * width * 4) + (x * 4);
            if (neighborIndex >= 0 && neighborIndex < data.length) {
                totalAlpha += data[neighborIndex + 3];
                count++;
            }
        }
    }

    return totalAlpha / count; // Average the alpha values of surrounding pixels
}

window.onload = () => {
    clientId = document.getElementById('clientId').value
    setInterval(() => {
        checkHung()
    }, 2000) // Check session activity every 2 seconds
    checkAndExecute()
}

window.startSession = () => {
    createSpeechRecognizer()
    if (document.getElementById('useLocalVideoForIdle').checked) {
        document.getElementById('startSession').disabled = true
        document.getElementById('configuration').hidden = true
        document.getElementById('microphone').disabled = false
        document.getElementById('stopSession').disabled = false
        document.getElementById('localVideo').hidden = false
        document.getElementById('remoteVideo').style.width = '0.1px'
        document.getElementById('chatHistory').hidden = false
        document.getElementById('latencyLog').hidden = false
        document.getElementById('showTypeMessage').disabled = false
        return
    }

    connectAvatar()
}

window.stopSpeaking = () => {
    document.getElementById('stopSpeaking').disabled = true

    fetch('/api/stopSpeaking', {
        method: 'POST',
        headers: {
            'ClientId': clientId
        },
        body: ''
    })
    .then(response => {
        if (response.ok) {
            console.log('Successfully stopped speaking.')
        } else {
            throw new Error(`Failed to stop speaking: ${response.status} ${response.statusText}`)
        }
    })
}

window.stopSession = () => {
    document.getElementById('startSession').disabled = false
    document.getElementById('microphone').disabled = true
    document.getElementById('stopSession').disabled = true
    document.getElementById('configuration').hidden = false
    document.getElementById('chatHistory').hidden = true
    document.getElementById('latencyLog').hidden = true
    document.getElementById('showTypeMessage').checked = false
    document.getElementById('showTypeMessage').disabled = true
    document.getElementById('userMessageBox').hidden = true
    if (document.getElementById('useLocalVideoForIdle').checked) {
        document.getElementById('localVideo').hidden = true
    }

    disconnectAvatar(true)
}

window.clearChatHistory = () => {
    fetch('/api/chat/clearHistory', {
        method: 'POST',
        headers: {
            'ClientId': clientId
        },
        body: ''
    })
    .then(response => {
        if (response.ok) {
            document.getElementById('chatHistory').innerHTML = ''
            document.getElementById('latencyLog').innerHTML = ''
        } else {
            throw new Error(`Failed to clear chat history: ${response.status} ${response.statusText}`)
        }
    })
}

window.microphone = () => {
    let microphoneButton = document.getElementById('microphone');
    let microphoneIcon = microphoneButton.querySelector('i');

    if (microphoneIcon.classList.contains('fa-microphone-slash')) {
        // Stop microphone
        microphoneButton.disabled = true;
        speechRecognizer.stopContinuousRecognitionAsync(
            () => {
                microphoneIcon.classList.remove('fa-microphone-slash');
                microphoneIcon.classList.add('fa-microphone');
                microphoneButton.disabled = false;
            }, (err) => {
                console.log("Failed to stop continuous recognition:", err);
                microphoneButton.disabled = false;
            });
        return;
    }

    if (document.getElementById('useLocalVideoForIdle').checked) {
        if (!sessionActive) {
            connectAvatar();
        }

        setTimeout(() => {
            document.getElementById('audioPlayer').play();
        }, 5000);
    } else {
        document.getElementById('audioPlayer').play();
    }

    microphoneButton.disabled = true;
    speechRecognizer.recognizing = async (s, e) => {
        if (isFirstRecognizingEvent && isSpeaking) {
            window.stopSpeaking();
            isFirstRecognizingEvent = false;
        }
    };

    speechRecognizer.recognized = async (s, e) => {
        if (e.result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
            let userQuery = e.result.text.trim();
            if (userQuery === '') {
                return;
            }

            let recognitionResultReceivedTime = new Date();
            let speechFinishedOffset = (e.result.offset + e.result.duration) / 10000;
            let sttLatency = recognitionResultReceivedTime - recognitionStartedTime - speechFinishedOffset;
            console.log(`STT latency: ${sttLatency} ms`);
            let latencyLogTextArea = document.getElementById('latencyLog');
            latencyLogTextArea.innerHTML += `STT latency: ${sttLatency} ms\n`;
            latencyLogTextArea.scrollTop = latencyLogTextArea.scrollHeight;

            // Auto stop microphone when a phrase is recognized, when it's not continuous conversation mode
            if (!document.getElementById('continuousConversation').checked) {
                microphoneButton.disabled = true;
                speechRecognizer.stopContinuousRecognitionAsync(
                    () => {
                        microphoneIcon.classList.remove('fa-microphone-slash');
                        microphoneIcon.classList.add('fa-microphone');
                        microphoneButton.disabled = false;
                    }, (err) => {
                        console.log("Failed to stop continuous recognition:", err);
                        microphoneButton.disabled = false;
                    });
            }

            let chatHistoryTextArea = document.getElementById('chatHistory');
            if (chatHistoryTextArea.innerHTML !== '' && !chatHistoryTextArea.innerHTML.endsWith('\n\n')) {
                chatHistoryTextArea.innerHTML += '\n\n';
            }

            chatHistoryTextArea.innerHTML += "User: " + userQuery + '\n\n';
            chatHistoryTextArea.scrollTop = chatHistoryTextArea.scrollHeight;

            handleUserQuery(userQuery);

            isFirstRecognizingEvent = true;
        }
    };

    recognitionStartedTime = new Date();
    speechRecognizer.startContinuousRecognitionAsync(
        () => {
            microphoneIcon.classList.remove('fa-microphone');
            microphoneIcon.classList.add('fa-microphone-slash');
            microphoneButton.disabled = false;
        }, (err) => {
            console.log("Failed to start continuous recognition:", err);
            microphoneButton.disabled = false;
        });
};

window.toggleChat = () => {
    const chatBox = document.getElementById("chat-box");
    chatBox.style.display = chatBox.style.display === "none" ? "block" : "none";
}

window.updateTypeMessageBox = () => {
    if (document.getElementById('showTypeMessage').checked) {
        document.getElementById('userMessageBox').hidden = false
        document.getElementById('userMessageBox').addEventListener('keyup', (e) => {
            if (e.key === 'Enter') {
                const userQuery = document.getElementById('userMessageBox').value
                if (userQuery !== '') {
                    let chatHistoryTextArea = document.getElementById('chatHistory')
                    if (chatHistoryTextArea.innerHTML !== '' && !chatHistoryTextArea.innerHTML.endsWith('\n\n')) {
                        chatHistoryTextArea.innerHTML += '\n\n'
                    }

                    chatHistoryTextArea.innerHTML += "User: " + userQuery.trim('\n') + '\n\n'
                    chatHistoryTextArea.scrollTop = chatHistoryTextArea.scrollHeight

                    if (isSpeaking) {
                        window.stopSpeaking()
                    }

                    handleUserQuery(userQuery.trim('\n'))
                    document.getElementById('userMessageBox').value = ''
                }
            }
        })
    } else {
        document.getElementById('userMessageBox').hidden = true
    }
}

window.updateLocalVideoForIdle = () => {
    if (document.getElementById('useLocalVideoForIdle').checked) {
        document.getElementById('showTypeMessageCheckbox').hidden = true
    } else {
        document.getElementById('showTypeMessageCheckbox').hidden = false
    }
}