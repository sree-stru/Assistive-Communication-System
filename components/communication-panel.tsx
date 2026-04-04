"use client";

import { useState } from "react";
import { MessageSquare, Plus, Space, Volume2, Trash2 } from "lucide-react";
import { GestureGuide } from "./gesture-guide";

interface CommunicationPanelProps {
  accumulatedText: string;
  onAddGesture: () => void;
  onAddSpace: () => void;
  onSpeak: () => void;
  onClear: () => void;
}

export function CommunicationPanel({
  accumulatedText,
  onAddGesture,
  onAddSpace,
  onSpeak,
  onClear,
}: CommunicationPanelProps) {
  const [isSpeaking, setIsSpeaking] = useState(false);

  const handleSpeak = () => {
    if (!accumulatedText.trim()) {
      return;
    }
    
    setIsSpeaking(true);
    onSpeak();
    
    // Reset speaking state after estimated duration
    setTimeout(() => setIsSpeaking(false), 2000);
  };

  const handleClear = () => {
    if (confirm("Are you sure you want to clear all text?")) {
      onClear();
    }
  };

  return (
    <section className="rounded-xl bg-[var(--muted)] p-5">
      <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold text-[var(--primary)]">
        <MessageSquare className="h-5 w-5" />
        Communication Output
      </h2>

      <textarea
        value={accumulatedText}
        readOnly
        placeholder="Recognized gestures will appear here..."
        className="min-h-[180px] w-full resize-y rounded-lg border-2 border-[var(--border)] bg-white p-4 text-lg focus:border-[var(--primary)] focus:outline-none"
      />

      <div className="mt-4 grid grid-cols-2 gap-3">
        <button
          onClick={onAddGesture}
          className="flex items-center justify-center gap-2 rounded-lg bg-[var(--primary)] px-4 py-3 font-semibold text-white shadow-md transition-all hover:-translate-y-0.5 hover:shadow-lg"
        >
          <Plus className="h-5 w-5" />
          Add Gesture
        </button>
        
        <button
          onClick={onAddSpace}
          className="flex items-center justify-center gap-2 rounded-lg bg-[var(--secondary)] px-4 py-3 font-semibold text-[var(--secondary-foreground)] shadow-md transition-all hover:-translate-y-0.5 hover:shadow-lg"
        >
          <Space className="h-5 w-5" />
          Add Space
        </button>
        
        <button
          onClick={handleSpeak}
          disabled={isSpeaking || !accumulatedText.trim()}
          className="flex items-center justify-center gap-2 rounded-lg bg-[var(--success)] px-4 py-3 font-semibold text-white shadow-md transition-all hover:-translate-y-0.5 hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0"
        >
          <Volume2 className="h-5 w-5" />
          {isSpeaking ? "Speaking..." : "Speak"}
        </button>
        
        <button
          onClick={handleClear}
          disabled={!accumulatedText}
          className="flex items-center justify-center gap-2 rounded-lg bg-[var(--destructive)] px-4 py-3 font-semibold text-white shadow-md transition-all hover:-translate-y-0.5 hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0"
        >
          <Trash2 className="h-5 w-5" />
          Clear
        </button>
      </div>

      <GestureGuide />
    </section>
  );
}
