import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import { evaluate } from './evaluation';
import MessageList from './components/MessageList';
import InputArea from './components/InputArea';
import SettingsModal from './components/SettingsModal';
import LoadingIndicator from './components/LoadingIndicator';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [files, setFiles] = useState([]);
  const [tools, setTools] = useState({});
  const [enabledTools, setEnabledTools] = useState(() => {
    const saved = localStorage.getItem('enabledTools');
    return saved ? JSON.parse(saved) : {};
  });
  const [showSettings, setShowSettings] = useState(false);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState(() => {
    return localStorage.getItem('systemPrompt') || 'You are a helpful assistant.';
  });
  const [theme, setTheme] = useState(() => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.className = savedTheme;
    return savedTheme;
  });
  const fileInputRef = useRef(null);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const runEvaluation = async () => {
    const results = await evaluate(sendMessage);
    setEvaluationResults(results);
  };

  useEffect(() => {
    localStorage.setItem('systemPrompt', systemPrompt);
  }, [systemPrompt]);

  useEffect(() => {
    localStorage.setItem('theme', theme);
    document.body.className = theme; // Apply theme class to body
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  useEffect(() => {
    const fetchTools = async () => {
      try {
        const response = await axios.get('http://localhost:8080/tools-api/get_tools');
        const fetchedTools = response.data || { langgraph: [], mcpo: [] };
        setTools(fetchedTools);

        const savedEnabled = JSON.parse(localStorage.getItem('enabledTools'));
        if (!savedEnabled) {
          const initialEnabled = {};
          fetchedTools.langgraph.forEach(module => {
            initialEnabled[module.name] = true;
          });
          fetchedTools.mcpo.forEach(server => {
            initialEnabled[server.name] = true;
          });
          setEnabledTools(initialEnabled);
          localStorage.setItem('enabledTools', JSON.stringify(initialEnabled));
        } else {
          setEnabledTools(savedEnabled);
        }

      } catch (error) {
        console.error('Error fetching tools:', error);
        setMessages([{ role: 'assistant', content: 'Error: Could not load tools from the tools-api.' }]);
      }
    };
    fetchTools();
  }, []);

  const openSettings = () => setShowSettings(true);
  const closeSettings = () => setShowSettings(false);

  const handleToggleModule = (moduleName) => {
    const updatedEnabled = { ...enabledTools, [moduleName]: !enabledTools[moduleName] };
    setEnabledTools(updatedEnabled);
    localStorage.setItem('enabledTools', JSON.stringify(updatedEnabled));
  };

  const handleToggleServer = (serverName) => {
    const updatedEnabled = { ...enabledTools, [serverName]: !enabledTools[serverName] };
    setEnabledTools(updatedEnabled);
    localStorage.setItem('enabledTools', JSON.stringify(updatedEnabled));
  };

  const handleFileChange = (event) => {
    const selectedFiles = Array.from(event.target.files);
    const newFiles = [];

    selectedFiles.forEach(file => {
      const reader = new FileReader();
      reader.onload = (e) => {
        newFiles.push({ name: file.name, content: e.target.result });
        if (newFiles.length === selectedFiles.length) {
          setFiles(prevFiles => [...prevFiles, ...newFiles]);
        }
      };
      reader.readAsText(file);
    });
    event.target.value = null;
  };

  const triggerFileUpload = () => {
    fileInputRef.current.click();
  };

  const removeFile = (fileNameToRemove) => {
    setFiles(prevFiles => prevFiles.filter(file => file.name !== fileNameToRemove));
  };

  const startNewChat = () => {
    setMessages([]);
    setFiles([]);
    setInput('');
  };

  const sendMessage = async () => {
    const trimmedInput = input.trim();
    if (trimmedInput === '' && files.length === 0) return;

    const userMessageForDisplay = { role: 'user', content: trimmedInput };
    const currentMessages = [...messages, userMessageForDisplay];
    setMessages(currentMessages);
    setIsLoading(true);

    let apiContent = trimmedInput;
    if (files.length > 0) {
      const fileContext = files.map(file =>
        `CONTEXT FROM FILE: ${file.name}\n\n---\n\n${file.content}`
      ).join('\n\n---\n\n');
      apiContent = `${fileContext}\n\n---\n\nQUESTION:\n${trimmedInput}`;
    }

    setInput('');
    setFiles([]);

    try {
      const enabledLangGraphModules = tools.langgraph
        .filter(module => enabledTools[module.name])
        .map(module => module.name);

      const enabledMcpoServers = tools.mcpo
        .filter(server => enabledTools[server.name])
        .map(server => server.name);

      const payload = {
        messages: [
          { role: 'system', content: systemPrompt },
          ...messages, 
          { role: 'user', content: apiContent }
        ],
        enabled_tools: {
          langgraph: enabledLangGraphModules,
          mcpo: enabledMcpoServers,
        }
      };

      console.log('Sending payload to http://localhost:8080/agent-service/chat:', payload);

      const response = await axios.post('http://localhost:8080/agent-service/chat', payload);

      const botMessage = { role: 'assistant', content: response.data.response };
      setMessages(prevMessages => [...prevMessages, botMessage]);

    } catch (error) {
      console.error('Error communicating with the agent service:', error);
      let errorMsg = 'Error: Could not connect to the agent service.';
      if (error.response && error.response.data) {
        if (typeof error.response.data === 'string') {
          errorMsg += `\n${error.response.data}`;
        } else if (error.response.data.error) {
          errorMsg += `\n${error.response.data.error}`;
        } else {
          errorMsg += `\n${JSON.stringify(error.response.data, null, 2)}`;
        }
      } else if (error.message) {
        errorMsg += `\n${error.message}`;
      }
      setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: errorMsg }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Agent</h1>
        <div className="header-buttons">
          <button onClick={toggleTheme} className="theme-toggle">
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
          <button onClick={startNewChat}>New Chat</button>
          <button onClick={openSettings}>Settings</button>
          <button onClick={runEvaluation}>Run Evaluation</button>
        </div>
      </header>
      <div className="chat-container" ref={chatContainerRef}>
        <MessageList messages={messages} />
        {isLoading && <LoadingIndicator />}
        <InputArea
          input={input}
          setInput={setInput}
          files={files}
          sendMessage={sendMessage}
          removeFile={removeFile}
          triggerFileUpload={triggerFileUpload}
          fileInputRef={fileInputRef}
          handleFileChange={handleFileChange}
        />
      </div>
      {evaluationResults && (
        <div className="evaluation-results">
          <h2>Evaluation Results</h2>
          <pre>{JSON.stringify(evaluationResults, null, 2)}</pre>
        </div>
      )}
      <SettingsModal
        showSettings={showSettings}
        closeSettings={closeSettings}
        tools={tools}
        enabledTools={enabledTools}
        handleToggleModule={handleToggleModule}
        handleToggleServer={handleToggleServer}
        systemPrompt={systemPrompt}
        setSystemPrompt={setSystemPrompt}
      />
    </div>
  );
}

export default App;