import React, { useState } from 'react';

function MessageList({ messages }) {
  const [expandedMessages, setExpandedMessages] = useState({});

  const toggleMessageExpansion = (index) => {
    setExpandedMessages(prev => ({ ...prev, [index]: !prev[index] }));
  };

  return (
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
  );
}

export default MessageList;
