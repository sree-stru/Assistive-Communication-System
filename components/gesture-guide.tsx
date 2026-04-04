import { BookOpen } from "lucide-react";

const GESTURE_MAPPINGS = [
  { gesture: "Fist", meaning: "YES", fingers: 0 },
  { gesture: "One Finger", meaning: "I", fingers: 1 },
  { gesture: "Two Fingers", meaning: "NEED", fingers: 2 },
  { gesture: "Three Fingers", meaning: "HELP", fingers: 3 },
  { gesture: "Four Fingers", meaning: "WAIT", fingers: 4 },
  { gesture: "Open Palm", meaning: "HELLO", fingers: 5 },
];

export function GestureGuide() {
  return (
    <div className="mt-5 rounded-lg border-2 border-[var(--border)] bg-white p-4">
      <h3 className="mb-3 flex items-center gap-2 font-semibold text-[var(--primary)]">
        <BookOpen className="h-5 w-5" />
        Gesture Guide
      </h3>
      <ul className="space-y-2">
        {GESTURE_MAPPINGS.map((item) => (
          <li
            key={item.meaning}
            className="flex items-center justify-between border-b border-[var(--border)] pb-2 last:border-0 last:pb-0"
          >
            <span className="text-[var(--muted-foreground)]">
              <strong className="text-[var(--foreground)]">{item.gesture}</strong>
              <span className="ml-2 text-sm">({item.fingers} fingers)</span>
            </span>
            <span className="rounded bg-[var(--primary)]/10 px-2 py-1 text-sm font-semibold text-[var(--primary)]">
              {item.meaning}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
