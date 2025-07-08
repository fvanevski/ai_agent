# AI Agent

This project is a conversational AI agent with a web-based UI. It uses a microservices architecture to handle different aspects of the system, including a language model, an ASR service, and a tool execution service.

## Architecture

The project is divided into two main parts: a `react-client` frontend and a `services` backend.

### Frontend

The frontend is a React application that provides a chat interface for interacting with the AI agent. It communicates with the backend via a proxy router.

### Backend

The backend consists of several microservices:

*   **`proxy-router`**: A pure reverse proxy that routes requests to the appropriate backend service. It is the single entry point for all frontend requests.
*   **`agent-service`**: The core of the AI agent. It uses LangGraph to create a conversational agent that can use tools to answer questions.
*   **`asr-api`**: An automatic speech recognition (ASR) service that uses a fine-tuned Whisper model to transcribe audio.
*   **`tools-api`**: A service that exposes a set of tools that can be used by the agent.
*   **`vllm-agent`**: A service that runs the language model.

## Getting Started

### Prerequisites

*   [Node.js](https://nodejs.org/)
*   [Python](https://www.python.org/)
*   [Docker](https://www.docker.com/)

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-username/ai-agent.git
    cd ai-agent
    ```

2.  Install the frontend dependencies:

    ```bash
    cd react-client
    npm install
    cd ..
    ```

3.  Install the backend dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  Start the backend services:

    ```bash
    docker-compose up -d
    ```

2.  Start the proxy router:

    ```bash
    python services/proxy-router/proxy_router.py
    ```

3.  Start the agent service:

    ```bash
    python services/agent-service/agent.py
    ```

4.  Start the frontend:

    ```bash
    cd react-client
    npm start
    ```

The application will be available at [http://localhost:3000](http://localhost:3000).

## Usage

Once the application is running, you can interact with the AI agent by typing messages in the chat window. You can also upload files and select which tools the agent should use.
