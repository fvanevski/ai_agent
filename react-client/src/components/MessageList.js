import React, { useState, useMemo } from 'react';

const ToolCallDisplay = ({ toolCall, toolResult }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const toggleExpansion = () => setIsExpanded(prev => !prev);

  const args = toolCall.function.arguments ? JSON.parse(toolCall.function.arguments) : {};
  
  let resultDisplay;
  try {
    // Try to parse content if it's a JSON string, for pretty printing
    const parsedContent = JSON.parse(toolResult.content);
    resultDisplay = JSON.stringify(parsedContent, null, 2);
  } catch (e) {
    // If it's not a valid JSON string, display as raw text
    resultDisplay = toolResult.content;
  }

  return (
    <div className="message tool">
      <div className="tool-header" onClick={toggleExpansion}>
        <span className={`arrow ${isExpanded ? 'down' : 'right'}`}></span>
        <span className="tool-icon">🛠️</span>
        Tool Call: {toolCall.function.name}
      </div>
      {isExpanded && (
        <div className="tool-details">
          <div className="tool-section">
            <h4>Arguments:</h4>
            <pre className="tool-content">{JSON.stringify(args, null, 2)}</pre>
          </div>
          <div className="tool-section">
            <h4>Result:</h4>
            <pre className="tool-content">{resultDisplay}</pre>
          </div>
        </div>
      )}
    </div>
  );
};


function MessageList({ messages }) {
  const [expandedThoughts, setExpandedThoughts] = useState({});

  const toggleThoughtExpansion = (index) => {
    setExpandedThoughts(prev => ({ ...prev, [index]: !prev[index] }));
  };

  const processedMessages = useMemo(() => {
    const groupedMessages = [];
    let i = 0;
    while (i < messages.length) {
        const msg = messages[i];

        if (msg.role === 'assistant' && msg.tool_calls?.length > 0) {
            const toolCallIds = msg.tool_calls.map(tc => tc.id);
            // Find all tool results that immediately follow this message
            let j = i + 1;
            const correspondingResults = [];
            while (j < messages.length && messages[j].role === 'tool' && toolCallIds.includes(messages[j].tool_call_id)) {
                correspondingResults.push(messages[j]);
                j++;
            }

            const callResultPairs = msg.tool_calls.map(call => ({
                call,
                result: correspondingResults.find(res => res.tool_call_id === call.id)
            })).filter(pair => pair.result); // Only include pairs where a result was found

            groupedMessages.push({
                type: 'tool_group',
                assistantMessage: msg,
                callResultPairs,
            });
            i = j; // Move index past the processed tool results
        } else {
            groupedMessages.push({ type: 'message', message: msg });
            i++;
        }
    }
    return groupedMessages;
  }, [messages]);

  return (
    <div className="message-list">
      {processedMessages.map((item, index) => {
        if (item.type === 'tool_group') {
          const { assistantMessage, callResultPairs } = item;
          const thinkContent = assistantMessage.content?.match(/<think>(.*?)<\/think>/s)?.[1].trim();

          return (
            <React.Fragment key={index}>
              {thinkContent && (
                <div className="message assistant">
                  <div className="thought-bubble">
                    <div className="thought-header" onClick={() => toggleThoughtExpansion(index)}>
                      <span className={`arrow ${expandedThoughts[index] ? 'down' : 'right'}`}></span>
                      Thinking...
                    </div>
                    {expandedThoughts[index] && (
                      <pre className="thought-content">{thinkContent}</pre>
                    )}
                  </div>
                </div>
              )}
              {callResultPairs.map(({ call, result }, toolIndex) => (
                <ToolCallDisplay key={`${index}-${toolIndex}`} toolCall={call} toolResult={result} />
              ))}
            </React.Fragment>
          );
        }
        
        // Default message rendering
        const msg = item.message;
        if (!msg) return null;

        const content = msg.content || '';
        const thinkMatch = content.match(/<think>(.*?)<\/think>/s);
        const thinkContent = thinkMatch ? thinkMatch[1].trim() : null;
        const visibleContent = content.replace(/<think>.*?<\/think>/s, '').trim();

        // Don't render empty messages that were only for thinking or tool calls
        if (!visibleContent && (thinkContent || msg.tool_calls)) {
            if (!thinkContent) return null;
        }
        
        if (msg.role === 'assistant' && !visibleContent && !thinkContent) return null;


        return (
          <div key={index} className={`message ${msg.role}`}>
            {thinkContent && (
              <div className="thought-bubble">
                <div className="thought-header" onClick={() => toggleThoughtExpansion(index)}>
                  <span className={`arrow ${expandedThoughts[index] ? 'down' : 'right'}`}></span>
                  Thinking...
                </div>
                {expandedThoughts[index] && (
                  <pre className="thought-content">{thinkContent}</pre>
                )}
              </div>
            )}
            {visibleContent && <pre style={{ fontFamily: 'inherit', margin: 0, whiteSpace: 'pre-wrap' }}>{visibleContent}</pre>}
          </div>
        );
      })}
    </div>
  );
}

export default MessageList;
