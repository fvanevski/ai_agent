import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import { evaluate } from './evaluation';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [files, setFiles] = useState([]);
  const [tools, setTools] = useState({}); // Changed to an object to group by type
  const [enabledTools, setEnabledTools] = useState(() => {
    // Load from localStorage
    const saved = localStorage.getItem('enabledTools');
    return saved ? JSON.parse(saved) : {};
  });
  const [showSettings, setShowSettings] = useState(false);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const fileInputRef = useRef(null);
  const [expandedServers, setExpandedServers] = useState({});
  const [expandedMessages, setExpandedMessages] = useState({});

  const runEvaluation = async () => {
    const results = await evaluate(sendMessage);
    setEvaluationResults(results);
  };

  // Fetch tools from the tools-api service
  useEffect(() => {
    const fetchTools = async () => {
      try {
        const response = await axios.get('http://localhost:9000/get_tools');
        const fetchedTools = response.data || { langgraph: [], mcpo: [] };
        setTools(fetchedTools);

        const savedEnabled = JSON.parse(localStorage.getItem('enabledTools'));
        if (!savedEnabled) {
          // Initialize enabled state if nothing is saved
          const initialEnabled = {};
          fetchedTools.langgraph.forEach(tool => {
            initialEnabled[tool.name] = true; // LangGraph tools are toggled individually
          });
          fetchedTools.mcpo.forEach(server => {
            initialEnabled[server.name] = true; // MCPO servers are toggled as a group
          });
          setEnabledTools(initialEnabled);
          localStorage.setItem('enabledTools', JSON.stringify(initialEnabled));
        } else {
          // Use saved settings
          setEnabledTools(savedEnabled);
        }

      } catch (error) {
        console.error('Error fetching tools:', error);
        setMessages([{ text: 'Error: Could not load tools from the tools-api.', sender: 'bot' }]);
      }
    };
    fetchTools();
  }, []);

  // Settings modal logic
  const openSettings = () => setShowSettings(true);
  const closeSettings = () => setShowSettings(false);

  const handleToggleTool = (toolName) => {
    const updatedEnabled = { ...enabledTools, [toolName]: !enabledTools[toolName] };
    setEnabledTools(updatedEnabled);
    localStorage.setItem('enabledTools', JSON.stringify(updatedEnabled));
  };

  const handleToggleServer = (serverName) => {
    const updatedEnabled = { ...enabledTools, [serverName]: !enabledTools[serverName] };
    setEnabledTools(updatedEnabled);
    localStorage.setItem('enabledTools', JSON.stringify(updatedEnabled));
  };

  const handleToggleServerExpansion = (serverName) => {
    setExpandedServers(prev => ({ ...prev, [serverName]: !prev[serverName] }));
  };

  const toggleMessageExpansion = (index) => {
    setExpandedMessages(prev => ({ ...prev, [index]: !prev[index] }));
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
      const payload = {
        messages: [...messages, { role: 'user', content: apiContent }],
      };

      console.log('Sending payload to http://localhost:8000/chat:', payload);

      const response = await axios.post('http://localhost:8000/chat', payload);

      const botMessage = { role: 'assistant', content: response.data.response };
      setMessages(prevMessages => [...prevMessages, botMessage]);

    } catch (error) {
      console.error('Error communicating with the orchestrator:', error);
      let errorMsg = 'Error: Could not connect to the orchestrator.';
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
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Local LLM Chat</h1>
        <button onClick={startNewChat} className="new-chat-btn">New Chat</button>
        <button onClick={openSettings} className="settings-btn" style={{ marginLeft: 8 }}>Settings</button>
        <button onClick={runEvaluation} className="evaluation-btn" style={{ marginLeft: 8 }}>Run Evaluation</button>
      </header>
      <div className="chat-container">
        <div className="message-list">
          {messages.map((msg, index) => {
            if (msg.role === 'tool') {
              return (
                <div key={index} className="message tool">
                  <div className="tool-call">Tool Call: {msg.name}</div>
                  <pre className="tool-content">{msg.content}</pre>
                </div>
              )
            }
            
            const thinkMatch = msg.content.match(/<think>(.*?)<\/think>/s);
            const thinkContent = thinkMatch ? thinkMatch[1].trim() : null;
            const visibleContent = msg.content.replace(/<think>.*?<\/think>/s, '').trim();

            return (
              <div key={index} className={`message ${msg.role}`}>
                {thinkContent && (
                  <div className="thought-bubble">
                    <div className="thought-header" onClick={() => toggleMessageExpansion(index)}>
                      <span className={`arrow ${expandedMessages[index] ? 'down' : 'right'}`}></span>
                      Thinking...
                    </div>
                    {expandedMessages[index] && (
                      <pre className="thought-content">{thinkContent}</pre>
                    )}
                  </div>
                )}
                <pre style={{ fontFamily: 'inherit', margin: 0, whiteSpace: 'pre-wrap' }}>{visibleContent}</pre>
              </div>
            )
          })}
        </div>
        {evaluationResults && (
          <div className="evaluation-results">
            <h2>Evaluation Results</h2>
            <pre>{JSON.stringify(evaluationResults, null, 2)}</pre>
          </div>
        )}
        <div className="input-area">
          {files.length > 0 && (
            <div className="file-list">
              {files.map((file, index) => (
                <div key={index} className="file-item">
                  <span>{file.name}</span>
                  <button onClick={() => removeFile(file.name)} className="remove-file-btn">×</button>
                </div>
              ))}
            </div>
          )}
          <div className="input-container">
            <input
              type="file"
              multiple
              ref={fileInputRef}
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            <button onClick={triggerFileUpload} className="upload-btn">📎</button>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Type your message or upload files..."
            />
            <button onClick={sendMessage}>Send</button>
          </div>
        </div>
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="modal-overlay" style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.3)', zIndex: 1000 }}>
          <div className="modal">
            <div className="modal-content">
              <h2>Tool Settings</h2>
              
              {tools.langgraph && tools.langgraph.length > 0 && (
                <div>
                  <h3>LangGraph Tools</h3>
                  <ul style={{ listStyle: 'none', padding: 0 }}>
                    {tools.langgraph.map(tool => (
                      <li key={tool.name} style={{ marginBottom: 8, display: 'flex', alignItems: 'center' }}>
                        <span style={{ flex: 1 }}>
                          <strong>{tool.name}</strong>: {tool.description}
                        </span>
                        <label className="switch">
                          <input
                            type="checkbox"
                            checked={enabledTools[tool.name] || false}
                            onChange={() => handleToggleTool(tool.name)}
                          />
                          <span className="slider round"></span>
                        </label>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {tools.mcpo && tools.mcpo.length > 0 && (
                <div>
                  <h3>MCPO Servers</h3>
                  {tools.mcpo.map(server => (
                    <div key={server.name} className="server-section" style={{ marginBottom: 16, border: '1px solid #ccc', padding: 10, borderRadius: 5 }}>
                      <h4 onClick={() => handleToggleServerExpansion(server.name)}>
                        <span className={`arrow ${expandedServers[server.name] ? 'down' : 'right'}`}></span>
                        {server.name}
                        <div style={{flex: 1}}></div>
                        <label className="switch">
                          <input
                            type="checkbox"
                            checked={enabledTools[server.name] || false}
                            onChange={() => handleToggleServer(server.name)}
                            onClick={(e) => e.stopPropagation()} // Prevent expansion when toggling
                          />
                          <span className="slider round"></span>
                        </label>
                      </h4>
                      {expandedServers[server.name] && enabledTools[server.name] && (
                        <ul style={{ listStyle: 'none', padding: 0, marginLeft: 40 }}>
                          {server.tools.map(tool => (
                            <li key={tool.name} style={{ marginBottom: 8 }}>
                              <strong>{tool.name}</strong>: {tool.description}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button onClick={closeSettings}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
