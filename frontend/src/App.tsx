import React, { useState, useEffect, useRef } from 'react';
import { Room, createLocalAudioTrack, RoomEvent, Track } from 'livekit-client';
import { Mic, Upload, Play, Loader2, CheckCircle2, XCircle } from 'lucide-react';
import './index.css';

const API_BASE = ""; // Since we mount at /, relative path works. 
// However, during vite dev we might need http://localhost:8000. 
// For simplicity in production/combined mode, relative is best.

const App: React.FC = () => {
  const [status, setStatus] = useState<string>('Ready to connect');
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [fileName, setFileName] = useState<string>('No file loaded');
  const [isConnecting, setIsConnecting] = useState<boolean>(false);
  const [timer, setTimer] = useState<number>(0);
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
          <h1>{isConnected ? "Live Session" : "Voice Explorer"}</h1>
          <p className="subtitle">{isConnected ? "AI Knowledge Active" : "Premium AI Knowledge Interface"}</p>
        </div>

        {!isConnected && (
          <div className="upload-section">
            <div className="file-input-wrapper">
              <label htmlFor="docUpload" className="upload-label">
                {isUploading ? <Loader2 className="animate-spin" /> : <Upload size={20} />}
                {isUploading ? "Processing..." : "Upload Document"}
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
          {isConnecting ? "Connecting..." : isConnected ? "End Call" : "Start Conversation"}
        </button>
      </div>
    </div>
  );
};

export default App;
