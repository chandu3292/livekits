import React, { useState, useEffect, useRef } from 'react';
import { Upload, Loader2, History, X, FileText, Play, Send, Trash2, MessageSquare, Mic, XCircle } from 'lucide-react';
import { Room, createLocalAudioTrack, RoomEvent, Track } from 'livekit-client';
import './index.css';

interface Persona {
  id: string;
  name: string;
  language: string;
  description: string;
}

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'speech' | 'history'>('chat');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [fileName, setFileName] = useState<string>('No file loaded');
  const [sessions, setSessions] = useState<any[]>([]);
  const [selectedSession, setSelectedSession] = useState<any | null>(null);
  const [transcript, setTranscript] = useState<any[]>([]);
  const [isLoadingTranscript, setIsLoadingTranscript] = useState(false);

  // Chat state
  const [chatMessages, setChatMessages] = useState<Array<{ role: string; text: string }>>([]);
  const [chatInput, setChatInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Persona state
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [selectedPersona, setSelectedPersona] = useState<string>('sophia');

  // Speech state
  const [status, setStatus] = useState<string>('Ready to connect');
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isConnecting, setIsConnecting] = useState<boolean>(false);
  const [timer, setTimer] = useState<number>(0);
  const roomRef = useRef<Room | null>(null);

  useEffect(() => {
    fetchPersonas();
  }, []);

  useEffect(() => {
    if (activeTab === 'history') {
      fetchSessions();
    }
  }, [activeTab]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // Timer for speech connection
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;
    if (isConnected) {
      interval = setInterval(() => setTimer(t => t + 1), 1000);
    } else {
      setTimer(0);
    }
    return () => { if (interval) clearInterval(interval); };
  }, [isConnected]);

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  const fetchPersonas = async () => {
    try {
      const res = await fetch("/api/personas");
      const data = await res.json();
      setPersonas(data.personas || []);
      setSelectedPersona(data.active_persona_id || 'sophia');
    } catch (err) {
      console.error("Failed to fetch personas", err);
    }
  };

  const handlePersonaChange = async (personaId: string) => {
    setSelectedPersona(personaId);
    try {
      await fetch("/api/set-persona", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ persona_id: personaId })
      });
    } catch (err) {
      console.error("Failed to set persona", err);
    }
  };

  const fetchSessions = async () => {
    try {
      const res = await fetch("/api/sessions");
      const data = await res.json();
      setSessions(data);
    } catch (err) {
      console.error("Failed to fetch sessions", err);
    }
  };

  const openSession = async (session: any) => {
    setSelectedSession(session);
    setIsLoadingTranscript(true);
    try {
      const res = await fetch(`/api/sessions/${session.id}/transcript`);
      const data = await res.json();
      setTranscript(data);
    } catch (err) {
      console.error("Failed to fetch transcript", err);
    } finally {
      setIsLoadingTranscript(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setFileName(`Uploading: ${file.name}`);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.status === "success") {
        let info = `Loaded: ${file.name}`;
        if (data.ocr) {
          info += ` (${data.chars_extracted} chars`;
          if (data.ocr.tables_found > 0) info += `, ${data.ocr.tables_found} tables`;
          info += ')';
        }
        setFileName(info);
      } else {
        setFileName(`Error: ${data.message || 'Upload failed'}`);
      }
    } catch (err) {
      console.error(err);
      setFileName("Upload Failed");
    } finally {
      setIsUploading(false);
    }
  };

  const sendMessage = async () => {
    const message = chatInput.trim();
    if (!message || isSending) return;

    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', text: message }]);
    setIsSending(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId })
      });
      const data = await res.json();

      if (data.status === "success") {
        setChatMessages(prev => [...prev, { role: 'assistant', text: data.response }]);
      } else {
        setChatMessages(prev => [...prev, { role: 'assistant', text: `Error: ${data.message}` }]);
      }
    } catch (err) {
      console.error(err);
      setChatMessages(prev => [...prev, { role: 'assistant', text: 'Failed to get response. Please try again.' }]);
    } finally {
      setIsSending(false);
      inputRef.current?.focus();
    }
  };

  const clearChat = async () => {
    setChatMessages([]);
    try {
      await fetch("/api/chat/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
      });
    } catch (err) {
      console.error(err);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Speech connection
  const startConversation = async () => {
    if (isConnected) return;

    setIsConnecting(true);
    setStatus("Getting token...");

    try {
      const res = await fetch(`/token?persona=${selectedPersona}`);
      const { token, url } = await res.json();

      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
      });

      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio) {
          const audio = track.attach();
          document.body.appendChild(audio);
        }
      });

      room.on(RoomEvent.Disconnected, () => {
        setIsConnected(false);
        setStatus("Ready to connect");
        roomRef.current = null;
      });

      await room.connect(url, token);
      roomRef.current = room;
      setIsConnected(true);
      setStatus("Connected");

      const mic = await createLocalAudioTrack();
      await room.localParticipant.publishTrack(mic);

      setStatus("Agent Listening...");
    } catch (err) {
      console.error(err);
      setStatus("Connection Failed");
    } finally {
      setIsConnecting(false);
    }
  };

  const disconnectRoom = async () => {
    if (roomRef.current) {
      await roomRef.current.disconnect();
      setIsConnected(false);
      setStatus("Ready to connect");
      roomRef.current = null;
    }
  };

  return (
    <div className="app-container">
      <div className="luxury-card chat-layout">
        <div className="header">
          <h1>DocQuery</h1>
          <div className="persona-selector">
            <select
              value={selectedPersona}
              onChange={e => handlePersonaChange(e.target.value)}
              disabled={isConnected}
            >
              {personas.map(p => (
                <option key={p.id} value={p.id}>
                  {p.name} — {p.description}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="tabs">
          <button
            className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            <MessageSquare size={18} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
            Chat
          </button>
          <button
            className={`tab ${activeTab === 'speech' ? 'active' : ''}`}
            onClick={() => setActiveTab('speech')}
          >
            <Mic size={18} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
            Speech
          </button>
          <button
            className={`tab ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            <History size={18} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
            History
          </button>
        </div>

        {activeTab === 'chat' ? (
          <div className="chat-container">
            {/* Upload section */}
            <div className="upload-section compact">
              <div className="file-input-wrapper">
                <label htmlFor="docUpload" className="upload-label compact">
                  {isUploading ? <Loader2 className="animate-spin" size={16} /> : <Upload size={16} />}
                  {isUploading ? "Processing..." : "Upload Document"}
                </label>
                <input
                  type="file"
                  id="docUpload"
                  className="file-input"
                  accept=".txt,.md,.py,.json,.pdf,.docx,.doc,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp"
                  onChange={handleFileUpload}
                  disabled={isUploading}
                />
              </div>
              <div className="file-name">{fileName}</div>
            </div>

            {/* Chat messages */}
            <div className="chat-messages">
              {chatMessages.length === 0 ? (
                <div className="chat-empty">
                  <MessageSquare size={48} opacity={0.2} />
                  <p>Start a conversation with the AI agent.</p>
                  <p className="hint">Upload documents for context-aware answers.</p>
                </div>
              ) : (
                chatMessages.map((msg, idx) => (
                  <div key={idx} className={`message ${msg.role}`}>
                    <div className="message-content">{msg.text}</div>
                  </div>
                ))
              )}
              {isSending && (
                <div className="message assistant">
                  <div className="message-content typing">
                    <span className="dot"></span>
                    <span className="dot"></span>
                    <span className="dot"></span>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Chat input */}
            <div className="chat-input-section">
              {chatMessages.length > 0 && (
                <button className="clear-chat-btn" onClick={clearChat} title="Clear chat">
                  <Trash2 size={16} />
                </button>
              )}
              <textarea
                ref={inputRef}
                className="chat-input"
                placeholder="Type a message... (Shift+Enter for new line)"
                value={chatInput}
                onChange={e => setChatInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
                disabled={isSending}
              />
              <button
                className="send-btn"
                onClick={sendMessage}
                disabled={isSending || !chatInput.trim()}
              >
                {isSending ? <Loader2 className="animate-spin" size={20} /> : <Send size={20} />}
              </button>
            </div>
          </div>
        ) : activeTab === 'speech' ? (
          <div className="speech-container">
            {/* Upload section */}
            <div className="upload-section compact">
              <div className="file-input-wrapper">
                <label htmlFor="docUploadSpeech" className="upload-label compact">
                  {isUploading ? <Loader2 className="animate-spin" size={16} /> : <Upload size={16} />}
                  {isUploading ? "Processing..." : "Upload Document"}
                </label>
                <input
                  type="file"
                  id="docUploadSpeech"
                  className="file-input"
                  accept=".txt,.md,.py,.json,.pdf,.docx,.doc,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp"
                  onChange={handleFileUpload}
                  disabled={isUploading}
                />
              </div>
              <div className="file-name">{fileName}</div>
            </div>

            <div className="speech-content">
              <div className="status-section">
                <div className="status-badge">
                  <div className={`status-dot ${isConnected ? 'active' : ''}`} />
                  <span>{status}</span>
                </div>

                {isConnected && (
                  <div className="call-overlay">
                    <div className="timer">{formatTime(timer)}</div>
                    <div className="visualizer-container">
                      {[...Array(16)].map((_, i) => (
                        <div
                          key={i}
                          className="bar"
                          style={{
                            height: `${Math.random() * 50 + 5}px`,
                            animation: `barPulse ${0.3 + Math.random() * 0.7}s infinite alternate`
                          }}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <button
                className={`action-button ${isConnected ? 'disconnect' : ''}`}
                onClick={isConnected ? disconnectRoom : startConversation}
                disabled={isConnecting}
              >
                {isConnecting ? (
                  <Loader2 className="animate-spin" />
                ) : isConnected ? (
                  <XCircle size={24} />
                ) : (
                  <Mic size={24} />
                )}
                {isConnecting ? "Connecting..." : isConnected ? "End Conversation" : "Start Conversation"}
              </button>
            </div>
          </div>
        ) : (
          <div className="history-list">
            {sessions.length === 0 ? (
              <p style={{ textAlign: 'center', opacity: 0.5 }}>No recorded sessions yet.</p>
            ) : (
              sessions.map((s) => (
                <div key={s.id} className="session-item" onClick={() => openSession(s)}>
                  <div className="session-info">
                    <h4>{s.user}</h4>
                    <p>{s.time}</p>
                  </div>
                  <div style={{ display: 'flex', gap: '10px' }}>
                    <FileText size={18} opacity={s.transcript_exists ? 1 : 0.2} />
                    <Play size={18} />
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {selectedSession && (
        <div className="transcript-modal" onClick={() => setSelectedSession(null)}>
          <div className="transcript-card" onClick={e => e.stopPropagation()}>
            <div className="transcript-header">
              <div>
                <h2 style={{ margin: 0, fontSize: '1.2rem' }}>{selectedSession.user}</h2>
                <span style={{ fontSize: '0.8rem', opacity: 0.5 }}>{selectedSession.time}</span>
              </div>
              <button className="close-btn" onClick={() => setSelectedSession(null)}>
                <X size={20} />
              </button>
            </div>

            <div className="transcript-body">
              {isLoadingTranscript ? (
                <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <Loader2 className="animate-spin" size={40} />
                </div>
              ) : transcript.length === 0 ? (
                <p style={{ textAlign: 'center', padding: '2rem', opacity: 0.5 }}>Transcript not available for this session.</p>
              ) : (
                transcript.map((line, idx) => (
                  <div key={idx} className={`message ${line.role || 'system'}`}>
                    <span className="ts">{line.time}</span>
                    <div>{line.text || line.raw}</div>
                  </div>
                ))
              )}
            </div>

            <div className="transcript-footer">
              <audio
                controls
                className="audio-player"
                src={`/sessions/${selectedSession.filename}`}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
