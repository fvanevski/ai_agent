import React, { useRef, useEffect } from 'react';

function InputArea({ input, setInput, files, sendMessage, removeFile, triggerFileUpload, fileInputRef, handleFileChange }) {
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  return (
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
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
          placeholder="Type your message or upload files..."
          rows="1"
        ></textarea>
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default InputArea;
