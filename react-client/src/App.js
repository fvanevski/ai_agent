import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import { evaluate } from './evaluation';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [files, setFiles] = useState([]);
  const [tools, setTools] = useState([]);
  const [showSettings, setShowSettings] = useState(false);
  const [mcpoServers, setMcpoServers] = useState(() => {
    // Load from localStorage or default to localhost
    const saved = localStorage.getItem('mcpoServers');
    return saved ? JSON.parse(saved) : ['http://localhost:9002'];
  });
  const [newServerUrl, setNewServerUrl] = useState('');
  const [testResults, setTestResults] = useState({});
  const [testing, setTesting] = useState({});
  const [evaluationResults, setEvaluationResults] = useState(null);
  const fileInputRef = useRef(null);

  const runEvaluation = async () => {
    const results = await evaluate(sendMessage);
    setEvaluationResults(results);
  };

  // Fetch tools from all configured MCP servers
  useEffect(() => {
    const fetchAllTools = async () => {
      let allTools = [];
      let anyError = false;
      for (const serverUrl of mcpoServers) {
        try {
          const response = await axios.get(`${serverUrl.replace(/\/$/, '')}/openapi.json`);
          const openapiSchema = response.data;
          // Attach serverUrl to each tool
          const processedTools = processOpenAPISchema(openapiSchema).map(tool => ({ ...tool, serverUrl }));
          allTools = allTools.concat(processedTools);
        } catch (error) {
          anyError = true;
          console.error(`Error fetching tools from ${serverUrl}:`, error);
        }
      }
      // Deduplicate tools by function name
      const uniqueTools = [];
      const seen = new Set();
      for (const tool of allTools) {
        if (tool.function && !seen.has(tool.function.name)) {
          uniqueTools.push(tool);
          seen.add(tool.function.name);
        }
      }
      setTools(uniqueTools);
      if (anyError && uniqueTools.length === 0) {
        setMessages([{ text: 'Error: Could not load tools from any MCP server. Please check your settings.', sender: 'bot' }]);
      }
    };
    fetchAllTools();
  }, [mcpoServers]);
  // Settings modal logic
  const openSettings = () => setShowSettings(true);
  const closeSettings = () => setShowSettings(false);

  const handleAddServer = () => {
    const url = newServerUrl.trim();
    if (!url || mcpoServers.includes(url)) return;
    const updated = [...mcpoServers, url];
    setMcpoServers(updated);
    localStorage.setItem('mcpoServers', JSON.stringify(updated));
    setNewServerUrl('');
  };

  const handleRemoveServer = (url) => {
    const updated = mcpoServers.filter(u => u !== url);
    setMcpoServers(updated);
    localStorage.setItem('mcpoServers', JSON.stringify(updated));
    setTestResults(prev => {
      const copy = { ...prev };
      delete copy[url];
      return copy;
    });
  };

  const handleTestServer = async (url) => {
    setTesting(prev => ({ ...prev, [url]: true }));
    try {
      const response = await axios.get(`${url.replace(/\/$/, '')}/openapi.json`, { timeout: 4000 });
      if (response.data && response.data.openapi) {
        setTestResults(prev => ({ ...prev, [url]: 'success' }));
      } else {
        setTestResults(prev => ({ ...prev, [url]: 'invalid' }));
      }
    } catch (e) {
      setTestResults(prev => ({ ...prev, [url]: 'fail' }));
    }
    setTesting(prev => ({ ...prev, [url]: false }));
  };

  const processOpenAPISchema = (schema) => {
    const processed = [];
    if (!schema || !schema.paths) return processed;

    for (const path in schema.paths) {
      for (const method in schema.paths[path]) {
        if (method.toLowerCase() === 'post') {
          const toolInfo = schema.paths[path][method];
          const functionName = toolInfo.operationId.replace(/_post$/, '').replace(/_/g, '-');
          const description = toolInfo.description || toolInfo.summary;
          const requestBody = toolInfo.requestBody;
          let parameters = { type: 'object', properties: {}, required: [] };

          if (requestBody && requestBody.content && requestBody.content['application/json']) {
            const schemaRef = requestBody.content['application/json'].schema['$ref'];
            if (schemaRef) {
              const schemaName = schemaRef.split('/').pop();
              const componentSchema = schema.components.schemas[schemaName];
              if (componentSchema) {
                parameters.properties = componentSchema.properties;
                parameters.required = componentSchema.required || [];
              }
            }
          }
          // Store the OpenAPI path for this tool
          processed.push({
            type: 'function',
            function: {
              name: functionName,
              description: description,
              parameters: parameters,
            },
            openapiPath: path,
          });
        }
      }
    }
    return processed;
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

    const systemPrompt = {
      role: 'system',
      content: "You are an expert AI assistant with the ability to execute tools in parallel. When a user's request involves multiple independent tasks, you should call the necessary tools in a single turn to maximize efficiency. For example, if asked to 'get the weather in New York and London', you should respond with two parallel calls to the `get_weather` tool. Your goal is to identify opportunities for parallel execution and use them whenever possible to provide a faster and more efficient response."
    };
    const fewShotExamples = [
      {
        "role": "user",
        "content": "What's the weather in New York and London?"
      },
      {
        "role": "assistant",
        "content": JSON.stringify([
          {
            "tool_name": "get_weather",
            "parameters": {
              "city": "New York"
            }
          },
          {
            "tool_name": "get_weather",
            "parameters": {
              "city": "London"
            }
          }
        ], null, 2)
      },
      {
        "role": "user",
        "content": "Summarize the following articles: [URL1] and [URL2]"
      },
      {
        "role": "assistant",
        "content": JSON.stringify([
          {
            "tool_name": "summarize_article",
            "parameters": {
              "url": "[URL1]"
            }
          },
          {
            "tool_name": "summarize_article",
            "parameters": {
              "url": "[URL2]"
            }
          }
        ], null, 2)
      }
    ];
    const apiMessages = [systemPrompt, ...fewShotExamples, ...messages.map(msg => ({
      role: msg.role,
      content: msg.content,
      tool_calls: msg.tool_calls
    })), { role: 'user', content: apiContent }];

    setInput('');
    setFiles([]);

    try {
      let payload;
      if (tools && tools.length > 0) {
        payload = {
          model: 'chat',
          messages: apiMessages,
          temperature: 0.7,
          tools: tools.map(({ serverUrl, ...rest }) => rest),
          tool_choice: 'auto',
        };
      } else {
        payload = {
          model: 'chat',
          messages: apiMessages,
          temperature: 0.7,
        };
      }

      console.log('Sending payload to http://localhost:8002/v1/chat/completions:', payload);

      let response = await axios.post('http://localhost:8002/v1/chat/completions', payload);

      let botMessage = response.data.choices[0].message;
      setMessages(prevMessages => [...prevMessages, botMessage]);

      if (botMessage.tool_calls && botMessage.tool_calls.length > 0) {
        const toolPromises = botMessage.tool_calls.map(async (toolCall) => {
          const toolName = toolCall.function.name;
          const toolArgs = JSON.parse(toolCall.function.arguments);
          // Find the tool definition and use its openapiPath
          const toolDef = tools.find(t => t.function && t.function.name === toolName);
          const serverUrl = toolDef && toolDef.serverUrl ? toolDef.serverUrl.replace(/\/$/, '') : '/mcpo';
          const openapiPath = toolDef && toolDef.openapiPath ? toolDef.openapiPath : `/${toolName}`;
          // Try both with and without trailing slash
          const endpoints = [openapiPath, openapiPath.endsWith('/') ? openapiPath : openapiPath + '/'];
          let lastError = null;
          for (const endpoint of endpoints) {
            try {
              console.log('Calling MCP tool:', {
                url: `${serverUrl}${endpoint}`,
                args: toolArgs,
                headers: { 'Content-Type': 'application/json' }
              });
              const toolResponse = await axios.post(`${serverUrl}${endpoint}`, toolArgs, {
                headers: { 'Content-Type': 'application/json' }
              });
              return {
                tool_call_id: toolCall.id,
                role: 'tool',
                name: toolName,
                content: JSON.stringify(toolResponse.data, null, 2),
              };
            } catch (toolError) {
              lastError = toolError;
              // If not last endpoint, try next
            }
          }
          // If both fail, log and return error
          console.error('All MCP tool endpoint attempts failed:', {
            endpoints: endpoints.map(e => `${serverUrl}${e}`),
            args: toolArgs,
            error: lastError
          });
          return {
            tool_call_id: toolCall.id,
            role: 'tool',
            name: toolName,
            content: `Error executing tool: ${lastError && lastError.message ? lastError.message : lastError}`,
          };
        });

        const toolResults = await Promise.all(toolPromises);
        setMessages(prevMessages => [...prevMessages, ...toolResults]);

        // vLLM expects tool results as an array of objects with name, tool_call_id, role, content
        const messagesWithToolResults = [...apiMessages, botMessage, ...toolResults];

        let payload2;
        if (tools && tools.length > 0) {
          payload2 = {
            model: 'chat',
            messages: messagesWithToolResults,
            temperature: 0.7,
            tools: tools.map(({ serverUrl, ...rest }) => rest),
            tool_choice: 'auto',
          };
        } else {
          payload2 = {
            model: 'chat',
            messages: messagesWithToolResults,
            temperature: 0.7,
          };
        }

        try {
          response = await axios.post('http://localhost:8002/v1/chat/completions', payload2);
          let finalBotMessage = response.data.choices[0].message;
          setMessages(prevMessages => [...prevMessages, finalBotMessage]);
        } catch (error) {
          console.error('Error during second LLM call:', error);
          setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: 'Error during second LLM call.' }]);
        }
      }
    } catch (error) {
      console.error('Error communicating with the LLM:', error);
      let errorMsg = 'Error: Could not connect to the LLM.';
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
            return (
              <div key={index} className={`message ${msg.role}`}>
                <pre style={{ fontFamily: 'inherit', margin: 0, whiteSpace: 'pre-wrap' }}>{msg.content}</pre>
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
        <div className="modal-overlay" style={{ position: 'fixed', top:0, left:0, right:0, bottom:0, background: 'rgba(0,0,0,0.3)', zIndex: 1000 }}>
          <div className="modal" style={{ background: '#fff', padding: 24, borderRadius: 8, maxWidth: 480, margin: '60px auto', position: 'relative' }}>
            <h2>MCP Server Settings</h2>
            <div style={{ marginBottom: 16 }}>
              <input
                type="text"
                value={newServerUrl}
                onChange={e => setNewServerUrl(e.target.value)}
                placeholder="Add MCP server URL (e.g. http://localhost:9002)"
                style={{ width: '70%' }}
              />
              <button onClick={handleAddServer} style={{ marginLeft: 8 }}>Add</button>
            </div>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              {mcpoServers.map((url, idx) => (
                <li key={url} style={{ marginBottom: 8, display: 'flex', alignItems: 'center' }}>
                  <span style={{ flex: 1 }}>{url}</span>
                  <button onClick={() => handleTestServer(url)} disabled={testing[url]} style={{ marginRight: 8 }}>
                    {testing[url] ? 'Testing...' : 'Test'}
                  </button>
                  {testResults[url] === 'success' && <span style={{ color: 'green', marginRight: 8 }}>✓</span>}
                  {testResults[url] === 'fail' && <span style={{ color: 'red', marginRight: 8 }}>✗</span>}
                  {testResults[url] === 'invalid' && <span style={{ color: 'orange', marginRight: 8 }}>Invalid</span>}
                  <button onClick={() => handleRemoveServer(url)} style={{ color: 'red' }}>Remove</button>
                </li>
              ))}
            </ul>
            <button onClick={closeSettings} style={{ marginTop: 16 }}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
