import React from 'react';

function InputArea({ input, setInput, files, sendMessage, removeFile, triggerFileUpload, fileInputRef, handleFileChange }) {
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
  );
}

export default InputArea;
