import { HandMetal } from "lucide-react";

export function Header() {
  return (
    <header className="bg-gradient-to-r from-[var(--primary)] to-[var(--accent)] px-6 py-8 text-center text-white">
      <div className="flex items-center justify-center gap-3">
        <HandMetal className="h-10 w-10" />
        <h1 className="text-3xl font-bold tracking-tight md:text-4xl">
          Assistive Communication System
        </h1>
      </div>
      <p className="mt-3 text-lg opacity-90">
        Real-time Hand Gesture Recognition for Speech and Hearing Impaired
      </p>
    </header>
  );
}
