# Transforming Digital Interactions with Hyper-Realistic Custom Avatars and Custom Neural Voices

This innovative solution combines Azure Text-to-Speech Custom Avatar Real-time API service and Custom Neural Voices to deliver hyper-realistic avatars with lifelike expressions and movements. Paired with advanced AI capabilities, these avatars enable seamless, human-like interactions tailored to diverse applications, from customer support to educational tools. By leveraging Retrieval Augmented Generation (RAG) using Azure OpenAI and Azure AI Search, the system ensures precise, contextually aware responses, redefining the way we engage and communicate in the digital age.

---

## Pre-requisites

Ensure the following Azure services are deployed before running this project:

1. **Azure Speech Service**:
   - For Text-to-Speech (TTS) and Speech-to-Text (STT) functionalities.
2. **Azure OpenAI Service**:
   - For natural language response generation using GPT models.
3. **Azure AI Search Service**: _(optional, if using your own data)_
   - For contextual data retrieval using the "Bring Your Own Data" feature of Azure OpenAI.
   - You can follow the instructions [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/use-your-data-quickstart?tabs=command-line%2Cjavascript-keyless%2Ctypescript-keyless%2Cpython-new&pivots=programming-language-studio).
4. **Azure Storage Account**: _(optional, if using your own data)_
   - To store customer-provided data for the search service.

---

## Setup Instructions

### Step 1: Clone the Repository
Clone the repository to your local environment:

```
git clone https://github.com/aadrikasingh/Azure-Text-To-Speech-Avatar.git
cd Azure-Text-To-Speech-Avatar
```

### Step 2: Install Dependencies
Install required Python packages using:

```
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a `.env` file in the project root and set the following environment variables:

**(Please ensure you begin with the `.env.sample` template)**

#### Azure Speech Configuration
```
SPEECH_REGION = "<Azure Speech Region, e.g. westus2>"
SPEECH_KEY = "<Azure Speech API Key>"
```

#### Avatar Configuration
```
AVATAR_CHARACTER="<Avatar Character Name>"
AVATAR_STYLE="<Avatar Style>"
IS_CUSTOM_AVATAR="<True/False>"
```

#### Neural Voice Configuration
```
TTS_VOICE="<Name of the TTS voice>"
CUSTOM_VOICE_ENDPOINT="<Optional: Endpoint for your Custom Neural Voice>"
PERSONAL_VOICE_SPEAKER_PROFILE="<Optional: Speaker profile ID for Personal Neural Voice>"
```

#### Azure OpenAI Configuration
```
AZURE_OPENAI_ENDPOINT="<Azure OpenAI Endpoint>"
AZURE_OPENAI_API_KEY="<Azure OpenAI API Key>"
AZURE_OPENAI_DEPLOYMENT_NAME="<Deployment Name>"
AZURE_OPENAI_SYSTEM_PROMPT="<System Prompt - update as needed>
```

#### Azure AI Search Configuration (Optional)
```
COGNITIVE_SEARCH_ENDPOINT="<Azure AI Search Endpoint>"
COGNITIVE_SEARCH_API_KEY="<Azure Search API Key>"
COGNITIVE_SEARCH_INDEX_NAME="<Search Index Name>"
```

#### Webpage Customization
For customizing the UI:
```
WEBPAGE_BACKGROUND_LANDSCAPE="<URL to Landscape Background>"
WEBPAGE_CHAT_FONTCOLOR_LANDSCAPE="#EEE"
BUTTON_COLOR_LANDSCAPE="#3E66BA"
BUTTON_HOVER_LANDSCAPE="#28a745"

WEBPAGE_BACKGROUND_PORTRAIT="<URL to Portrait Background>"
WEBPAGE_CHAT_FONTCOLOR_PORTRAIT="#EEE"
BUTTON_COLOR_PORTRAIT="#3E66BA"
BUTTON_HOVER_PORTRAIT="#28a745"
```

#### Set the welcome message
Please change line 267 & 268 in static/js/chat.js file
    
---    

## Running the Application

1. **Start the Flask Application**:

   Run the following command to launch the web app:
   ```
   python -m flask run -h 0.0.0.0 -p 5000
   ```

2. **Access the Web Interface (Landscape Orientation)**:

   Open your browser and navigate to:
   ```
   http://localhost:5000/chat
   ```

3. **Access the Web Interface (Portrait Orientation)**:

   Open your browser and navigate to:
   ```
   http://localhost:5000/portrait
   ```

4. **Initialize the Avatar Session**:
   - Click the first button **(Start Avatar Session)** to establish a connection with Azure TTS Avatar services.
   - If successful, you will see a live avatar video.

4. **Interact with the Avatar**:
   - Click the second button **(Start Microphone)** to enable speech input (ensure you allow microphone access in your browser).
   - Speak or type queries (with the **Chat** button)
   - The avatar will respond with synchronized audio and video.

---

## Additional Features

- **Interrupt Speech**:
  Use the **"Stop Speaking"** button to halt the avatar mid-sentence.

- **Clear Chat History**:
  Reset the session by clicking the **"Clear Chat History"** button.

- **Close Avatar Session**:
  End the avatar interaction with the **"Close Avatar Session"** button.

---

## Screenshots

### Landscape Mode
![Landscape Mode](https://github.com/aadrikasingh/Azure-Text-To-Speech-Avatar/blob/main/assets/landscape.png?raw=true)

### Portrait Mode
![Portrait Mode](https://github.com/aadrikasingh/Azure-Text-To-Speech-Avatar/blob/main/assets/portrait.png?raw=true)

---

## Adaptation
This implementation is adapted from the sample tutorial code provided by Microsoft. For more details, refer to the [original tutorial](https://github.com/Azure-Samples/cognitive-services-speech-sdk/tree/master/samples/js/browser/avatar).

---