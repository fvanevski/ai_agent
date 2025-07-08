import React, { useState } from 'react';

function SettingsModal({ showSettings, closeSettings, tools, enabledTools, handleToggleModule, handleToggleServer }) {
  const [expandedServers, setExpandedServers] = useState({});
  const [expandedModules, setExpandedModules] = useState({});

  const handleToggleServerExpansion = (serverName) => {
    setExpandedServers(prev => ({ ...prev, [serverName]: !prev[serverName] }));
  };

  const handleToggleModuleExpansion = (moduleName) => {
    setExpandedModules(prev => ({ ...prev, [moduleName]: !prev[moduleName] }));
  };

  if (!showSettings) {
    return null;
  }

  return (
    <div className="modal-overlay" style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.3)', zIndex: 1000 }}>
      <div className="modal">
        <div className="modal-content">
          <h2>Tool Settings</h2>
          
          {tools.langgraph && tools.langgraph.length > 0 && (
            <div>
              <h3>LangGraph Tools</h3>
              {tools.langgraph.map(module => (
                <div key={module.name} className="server-section" style={{ marginBottom: 16, border: '1px solid #ccc', padding: 10, borderRadius: 5 }}>
                  <h4 onClick={() => handleToggleModuleExpansion(module.name)}>
                    <span className={`arrow ${expandedModules[module.name] ? 'down' : 'right'}`}></span>
                    {module.name}
                    <div style={{flex: 1}}></div>
                    <label className="switch">
                      <input
                        type="checkbox"
                        checked={enabledTools[module.name] || false}
                        onChange={() => handleToggleModule(module.name)}
                        onClick={(e) => e.stopPropagation()}
                      />
                      <span className="slider round"></span>
                    </label>
                  </h4>
                  {expandedModules[module.name] && enabledTools[module.name] && (
                    <ul style={{ listStyle: 'none', padding: 0, marginLeft: 40 }}>
                      {module.tools.map(tool => (
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
  );
}

export default SettingsModal;
