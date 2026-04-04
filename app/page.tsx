"use client";

import { useState } from "react";
import { Header } from "@/components/header";
import { VideoSection } from "@/components/video-section";
import { CommunicationPanel } from "@/components/communication-panel";
import { Footer } from "@/components/footer";

export default function Home() {
  const [currentGesture, setCurrentGesture] = useState("No hand detected");
  const [accumulatedText, setAccumulatedText] = useState("");

  const handleAddGesture = () => {
    if (currentGesture !== "No hand detected") {
      setAccumulatedText((prev) =>
        prev ? `${prev} ${currentGesture}` : currentGesture
      );
    }
  };

  const handleAddSpace = () => {
    if (accumulatedText && !accumulatedText.endsWith(" ")) {
      setAccumulatedText((prev) => `${prev} `);
    }
  };

  const handleSpeak = () => {
    if (accumulatedText.trim() && "speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance(accumulatedText);
      speechSynthesis.speak(utterance);
    }
  };

  const handleClear = () => {
    setAccumulatedText("");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[var(--primary)] to-[var(--accent)] p-4 md:p-6">
      <div className="mx-auto max-w-7xl overflow-hidden rounded-2xl bg-[var(--card)] shadow-2xl">
        <Header />
        
        <main className="grid gap-6 p-6 lg:grid-cols-2">
          <VideoSection
            currentGesture={currentGesture}
            setCurrentGesture={setCurrentGesture}
          />
          <CommunicationPanel
            accumulatedText={accumulatedText}
            onAddGesture={handleAddGesture}
            onAddSpace={handleAddSpace}
            onSpeak={handleSpeak}
            onClear={handleClear}
          />
        </main>
        
        <Footer />
      </div>
    </div>
  );
}
