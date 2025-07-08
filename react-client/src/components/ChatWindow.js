import React from 'react';
import MessageList from './MessageList';
import InputArea from './InputArea';

function ChatWindow({ messages, files, input, setInput, setFiles, sendMessage, removeFile, triggerFileUpload, fileInputRef, handleFileChange }) {
  return (
    <div className="chat-container">
      <MessageList messages={messages} />
      <InputArea
        input={input}
        setInput={setInput}
        files={files}
        setFiles={setFiles}
        sendMessage={sendMessage}
        removeFile={removeFile}
        triggerFileUpload={triggerFileUpload}
        fileInputRef={fileInputRef}
        handleFileChange={handleFileChange}
      />
    </div>
  );
}

export default ChatWindow;
