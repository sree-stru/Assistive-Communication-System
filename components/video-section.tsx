"use client";

import { useEffect, useRef, useState } from "react";
import { Camera, CameraOff, Hand } from "lucide-react";

interface VideoSectionProps {
  currentGesture: string;
  setCurrentGesture: (gesture: string) => void;
}

const GESTURES = ["HELLO", "YES", "I", "NEED", "HELP", "WAIT"];

export function VideoSection({ currentGesture, setCurrentGesture }: VideoSectionProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let gestureInterval: NodeJS.Timeout;
    
    if (isStreaming) {
      // Simulate gesture detection for demo purposes
      gestureInterval = setInterval(() => {
        const randomGesture = GESTURES[Math.floor(Math.random() * GESTURES.length)];
        setCurrentGesture(randomGesture);
      }, 2000);
    }
    
    return () => {
      if (gestureInterval) clearInterval(gestureInterval);
    };
  }, [isStreaming, setCurrentGesture]);

  const startCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch {
      setError("Unable to access camera. Please check permissions.");
      setCurrentGesture("No hand detected");
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
      setCurrentGesture("No hand detected");
    }
  };

  return (
    <section className="rounded-xl bg-[var(--muted)] p-5">
      <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-[var(--primary)]">
        <Camera className="h-5 w-5" />
        Live Camera Feed
      </h2>

      <div className="relative aspect-video overflow-hidden rounded-lg bg-[var(--foreground)] shadow-lg">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-full w-full object-cover"
        />
        
        {!isStreaming && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-[var(--foreground)]/90 text-white">
            <CameraOff className="h-16 w-16 opacity-50" />
            <p className="text-lg">Camera is off</p>
          </div>
        )}
      </div>

      <div className="mt-4 flex gap-3">
        {!isStreaming ? (
          <button
            onClick={startCamera}
            className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-[var(--primary)] px-4 py-3 font-semibold text-white transition-all hover:opacity-90"
          >
            <Camera className="h-5 w-5" />
            Start Camera
          </button>
        ) : (
          <button
            onClick={stopCamera}
            className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-[var(--destructive)] px-4 py-3 font-semibold text-white transition-all hover:opacity-90"
          >
            <CameraOff className="h-5 w-5" />
            Stop Camera
          </button>
        )}
      </div>

      {error && (
        <p className="mt-3 text-sm text-[var(--destructive)]">{error}</p>
      )}

      <div className="mt-5 rounded-lg border-2 border-[var(--primary)] bg-white p-4 text-center">
        <h3 className="mb-2 flex items-center justify-center gap-2 text-sm font-medium text-[var(--muted-foreground)]">
          <Hand className="h-4 w-4" />
          Current Gesture
        </h3>
        <p className="rounded-lg bg-[var(--primary)]/10 px-4 py-3 text-2xl font-bold text-[var(--primary)]">
          {currentGesture}
        </p>
      </div>
    </section>
  );
}
