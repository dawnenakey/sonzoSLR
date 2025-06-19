export function formatTime(seconds) {
  if (seconds === null || seconds === undefined) return '00:00:00';
  
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  
  const pad = (num) => num.toString().padStart(2, '0');
  
  return `${pad(h)}:${pad(m)}:${pad(s)}${ms ? '.' + pad(ms) : ''}`;
}

export function parseTime(timeString) {
  if (!timeString) return 0;
  
  const parts = timeString.split(':');
  let seconds = 0;
  
  if (parts.length === 3) {
    // Handle hours:minutes:seconds format
    const [hours, minutes, secondsPart] = parts;
    seconds = parseInt(hours) * 3600 + parseInt(minutes) * 60;
    
    // Handle potential decimal seconds (e.g., "05.43")
    if (secondsPart.includes('.')) {
      const [wholeSec, fractionSec] = secondsPart.split('.');
      seconds += parseInt(wholeSec) + parseFloat(`0.${fractionSec}`);
    } else {
      seconds += parseInt(secondsPart);
    }
  }
  
  return seconds;
}