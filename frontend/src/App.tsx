import React, { useState, useEffect, useRef } from 'react';
import { Room, createLocalAudioTrack, RoomEvent, Track } from 'livekit-client';
import { Mic, Upload, Play, Loader2, CheckCircle2, XCircle, History, Video, X, FileText, Download } from 'lucide-react';
import './index.css';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'live' | 'history'>('live');
  const [status, setStatus] = useState<string>('Ready to connect');
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [fileName, setFileName] = useState<string>('No file loaded');
  const [isConnecting, setIsConnecting] = useState<boolean>(false);
  const [timer, setTimer] = useState<number>(0);
  const [sessions, setSessions] = useState<any[]>([]);
  const [selectedSession, setSelectedSession] = useState<any | null>(null);
  const [transcript, setTranscript] = useState<any[]>([]);
  const [isLoadingTranscript, setIsLoadingTranscript] = useState(false);

  const roomRef = useRef<Room | null>(null);
  const timerIntervalRef = useRef<any>(null);

  useEffect(() => {
    if (isConnected) {
      timerIntervalRef.current = setInterval(() => {
        setTimer(prev => prev + 1);
      }, 1000);
    } else {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      setTimer(0);
    }
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    };
  }, [isConnected]);

  useEffect(() => {
    if (activeTab === 'history') {
      fetchSessions();
    }
  }, [activeTab]);

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

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
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
        setFileName(`✅ Loaded: ${file.name}`);
      } else {
        setFileName("❌ Error uploading");
      }
    } catch (err) {
      console.error(err);
      setFileName("❌ Upload Failed");
    } finally {
      setIsUploading(false);
    }
  };

  const startConversation = async () => {
    if (isConnected) return;

    setIsConnecting(true);
    setStatus("Getting token…");

    try {
      const res = await fetch("/token");
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
      setIsConnecting(false);
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
      <div className="luxury-card">
        <div className="header">
          <h1>{isConnected ? "Live Session" : "LiveKit Agents"}</h1>
          <p className="subtitle">Multimodal Intelligence Interface</p>
        </div>

        {!isConnected && (
          <div className="tabs">
            <button
              className={`tab ${activeTab === 'live' ? 'active' : ''}`}
              onClick={() => setActiveTab('live')}
            >
              <Video size={18} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
              Call
            </button>
            <button
              className={`tab ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => setActiveTab('history')}
            >
              <History size={18} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
              History
            </button>
          </div>
        )}

        {activeTab === 'live' || isConnected ? (
          <>
            {!isConnected && (
              <div className="upload-section">
                <div className="file-input-wrapper">
                  <label htmlFor="docUpload" className="upload-label">
                    {isUploading ? <Loader2 className="animate-spin" /> : <Upload size={20} />}
                    {isUploading ? "Processing..." : "Context for Agent"}
                  </label>
                  <input
                    type="file"
                    id="docUpload"
                    className="file-input"
                    accept=".txt,.md,.py,.json,.pdf"
                    onChange={handleFileUpload}
                    disabled={isUploading}
                  />
                </div>
                <div className="file-name">{fileName}</div>
              </div>
            )}

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
                          animation: `pulse ${0.3 + Math.random() * 0.7}s infinite alternate`
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
          </>
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
