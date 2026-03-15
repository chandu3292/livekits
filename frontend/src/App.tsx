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
  const [chatMessages, setChatMessages] = useState<Array<{ role: string; text: string; time?: string }>>([]);
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

    const timeString = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', text: message, time: timeString }]);
    setIsSending(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId })
      });
      const data = await res.json();

      const resTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      if (data.status === "success") {
        setChatMessages(prev => [...prev, { role: 'assistant', text: data.response, time: resTime }]);
      } else {
        setChatMessages(prev => [...prev, { role: 'assistant', text: `Error: ${data.message}`, time: resTime }]);
      }
    } catch (err) {
      console.error(err);
      const errTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      setChatMessages(prev => [...prev, { role: 'assistant', text: 'Failed to get response. Please try again.', time: errTime }]);
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
    if (isConnecting) return;

    // Clean up any previous room first
    if (roomRef.current) {
      await roomRef.current.disconnect();
      roomRef.current = null;
    }

    setIsConnecting(true);
    setIsConnected(false);
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
          audio.volume = 1.0;
          audio.setAttribute('autoplay', 'true');
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
      {/* Left Navigation Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>DocQuery</h1>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            <MessageSquare size={18} />
            <span>Chat</span>
          </button>
          <button
            className={`nav-item ${activeTab === 'speech' ? 'active' : ''}`}
            onClick={() => setActiveTab('speech')}
          >
            <Mic size={18} />
            <span>Speech</span>
          </button>
          <button
            className={`nav-item ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            <History size={18} />
            <span>History</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          <span className="persona-label">Agent Persona</span>
          <div className="persona-selector">
            <select
              value={selectedPersona}
              onChange={e => handlePersonaChange(e.target.value)}
              disabled={isConnected}
            >
              {personas.map(p => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </aside>

      {/* Main Top-Level Area */}
      <main className="main-content">
        {/* Universal Top Bar for Context / Document Status */}
        {(activeTab === 'chat' || activeTab === 'speech') && (
          <header className="top-bar">
            <div className="top-bar-title">
              {activeTab === 'chat' ? 'Text Conversation' : 'Voice Interaction'}
            </div>
            <div className="upload-module">
              <span className="file-status" title={fileName}>{fileName}</span>
              <input
                type="file"
                id="universalUpload"
                className="file-input"
                accept=".txt,.md,.py,.json,.pdf,.docx,.doc,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp"
                onChange={handleFileUpload}
                disabled={isUploading}
              />
              <label htmlFor="universalUpload" className="upload-btn">
                {isUploading ? <Loader2 className="animate-spin" size={14} /> : <Upload size={14} />}
                <span>{isUploading ? "Uploading..." : "Add Context"}</span>
              </label>
            </div>
          </header>
        )}

        {/* Dynamic Views */}
        <div className="view-container">
          {activeTab === 'chat' ? (
            <div className="chat-view">
              <div className="chat-messages">
                {chatMessages.length === 0 ? (
                  <div className="chat-empty">
                    <div className="empty-icon"><MessageSquare size={32} /></div>
                    <h2>How can I help you?</h2>
                    <p>Start a conversation or upload context above.</p>
                  </div>
                ) : (
                  <div className="messages-wrapper">
                    {chatMessages.map((msg, idx) => (
                      <div key={idx} className={`message-row ${msg.role}`}>
                        <div className="message-bubble">
                          <div className="message-content">{msg.text}</div>
                          {msg.time && <span className="message-time">{msg.time}</span>}
                        </div>
                      </div>
                    ))}
                    {isSending && (
                      <div className="message-row assistant">
                        <div className="message-bubble typing-bubble">
                          <span className="dot"></span>
                          <span className="dot"></span>
                          <span className="dot"></span>
                        </div>
                      </div>
                    )}
                    <div ref={chatEndRef} />
                  </div>
                )}
              </div>

              <div className="chat-input-area">
                <div className="chat-input-wrapper">
                  {chatMessages.length > 0 && (
                    <button className="clear-btn" onClick={clearChat} title="Clear conversation">
                      <Trash2 size={18} />
                    </button>
                  )}
                  <textarea
                    ref={inputRef}
                    className="chat-input"
                    placeholder="Message DocQuery..."
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
                    {isSending ? <Loader2 className="animate-spin" size={18} /> : <Send size={18} />}
                  </button>
                </div>
                <div className="input-disclaimer">
                  AI can occasionally generate incorrect information. Verify critical data.
                </div>
              </div>
            </div>
          ) : activeTab === 'speech' ? (
            <div className="speech-view">
              <div className="speech-centerpiece">
                <div className="status-indicator">
                  <div className={`status-dot ${isConnected ? 'active' : ''}`} />
                  <span className="status-text">{status}</span>
                </div>

                {isConnected && (
                  <div className="active-call-ui">
                    <div className="timer-display">{formatTime(timer)}</div>
                    <div className="audio-visualizer">
                      {[...Array(24)].map((_, i) => (
                        <div
                          key={i}
                          className="wave-bar"
                          style={{
                            height: `${Math.random() * 60 + 10}px`,
                            animation: `wavePulse ${0.3 + Math.random() * 0.5}s infinite alternate`
                          }}
                        />
                      ))}
                    </div>
                  </div>
                )}

                <button
                  className={`call-action-btn ${isConnected ? 'end-call' : 'start-call'}`}
                  onClick={isConnected ? disconnectRoom : startConversation}
                  disabled={isConnecting}
                >
                  {isConnecting ? (
                    <Loader2 className="animate-spin" size={24} />
                  ) : isConnected ? (
                    <XCircle size={24} />
                  ) : (
                    <Mic size={24} />
                  )}
                  <span>{isConnecting ? "Connecting..." : isConnected ? "End Conversation" : "Start Voice Session"}</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="history-view">
              <div className="history-header">
                <h2>Session History</h2>
                <p>Review past voice interactions and generated transcripts.</p>
              </div>

              <div className="history-grid">
                {sessions.length === 0 ? (
                  <div className="history-empty">No recorded sessions yet.</div>
                ) : (
                  sessions.map((s) => (
                    <div key={s.id} className="history-card" onClick={() => openSession(s)}>
                      <div className="hc-primary">
                        <h3>{s.user}</h3>
                        <span className="hc-time">{s.time}</span>
                      </div>
                      <div className="hc-actions">
                        <FileText size={18} className={s.transcript_exists ? 'active-icon' : 'dim-icon'} />
                        <Play size={18} className="active-icon" />
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Transcript Modal overlay */}
      {selectedSession && (
        <div className="modal-overlay" onClick={() => setSelectedSession(null)}>
          <div className="modal-container" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div className="mh-info">
                <h2>{selectedSession.user}</h2>
                <span>{selectedSession.time}</span>
              </div>
              <button className="modal-close" onClick={() => setSelectedSession(null)}>
                <X size={24} />
              </button>
            </div>

            <div className="modal-body">
              {isLoadingTranscript ? (
                <div className="modal-loader">
                  <Loader2 className="animate-spin" size={40} />
                </div>
              ) : transcript.length === 0 ? (
                <div className="modal-empty">Transcript not available for this session.</div>
              ) : (
                <div className="transcript-list">
                  {transcript.map((line, idx) => (
                    <div key={idx} className={`transcript-row ${line.role || 'system'}`}>
                      <div className="tr-time">{line.time}</div>
                      <div className="tr-content">{line.text || line.raw}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="modal-footer">
              <audio
                controls
                className="modal-audio"
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
