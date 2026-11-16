const INTERVAL_MS = 4;

export function startEventLoopProbe() {
  let last = performance.now();
  let maxDelayMs = 0;
  const observe = () => {
    const now = performance.now();
    maxDelayMs = Math.max(maxDelayMs, now - last - INTERVAL_MS);
    last = now;
  };
  const timer = setInterval(observe, INTERVAL_MS);
  let stopped = false;

  return () => {
    if (!stopped) {
      stopped = true;
      clearInterval(timer);
      // A synchronous computation can finish before the timer gets its first turn.
      observe();
    }
    return maxDelayMs;
  };
}
