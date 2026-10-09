export interface TimestampedVector {
  x: number;
  y: number;
  z: number;
  timestampNs: number;
}

const MAX_SYNC_SKEW_NS: number = 20_000_000;

export function normalizeTimestampNs(timestamp: number | undefined): number {
  if (timestamp === undefined || timestamp <= 0) {
    return Date.now() * 1000000;
  }
  return timestamp;
}

export function pushToRingBuffer(buffer: TimestampedVector[], sample: TimestampedVector, maxSize: number): void {
  buffer.push(sample);
  if (buffer.length > maxSize) {
    buffer.shift();
  }
}

/**
 * 在均匀时间网格上线性插值，acc/gyro 共用同一 target 时刻。
 * 用于实时 100Hz 重采样输出。
 */
export function interpolateAtTimestamp(buffer: TimestampedVector[], targetNs: number): TimestampedVector | null {
  if (buffer.length === 0) {
    return null;
  }

  let prev: TimestampedVector | undefined = undefined;
  let next: TimestampedVector | undefined = undefined;

  for (let i = 0; i < buffer.length; i++) {
    const sample = buffer[i];
    if (sample.timestampNs <= targetNs) {
      if (!prev || sample.timestampNs >= prev.timestampNs) {
        prev = sample;
      }
    }
    if (sample.timestampNs >= targetNs) {
      if (!next || sample.timestampNs <= next.timestampNs) {
        next = sample;
      }
    }
  }

  if (prev && prev.timestampNs === targetNs) {
    return {
      x: prev.x,
      y: prev.y,
      z: prev.z,
      timestampNs: targetNs,
    };
  }

  if (prev && next && next.timestampNs > prev.timestampNs) {
    const ratio = (targetNs - prev.timestampNs) / (next.timestampNs - prev.timestampNs);
    return {
      x: prev.x + ratio * (next.x - prev.x),
      y: prev.y + ratio * (next.y - prev.y),
      z: prev.z + ratio * (next.z - prev.z),
      timestampNs: targetNs,
    };
  }

  const nearest = prev ?? next;
  if (nearest && Math.abs(nearest.timestampNs - targetNs) <= MAX_SYNC_SKEW_NS) {
    return {
      x: nearest.x,
      y: nearest.y,
      z: nearest.z,
      timestampNs: targetNs,
    };
  }

  return null;
}

/**
 * 在有序时间序列上按固定间隔重采样（线性插值，游标推进 O(n+m)）。
 */
export function resampleUniform(
  buffer: TimestampedVector[],
  startNs: number,
  endNs: number,
  intervalNs: number,
): TimestampedVector[] {
  const result: TimestampedVector[] = [];
  if (buffer.length === 0 || intervalNs <= 0 || endNs < startNs) {
    return result;
  }

  let index = 0;
  for (let targetNs = startNs; targetNs <= endNs; targetNs += intervalNs) {
    const sample = sampleAtWithCursor(buffer, targetNs, index);
    if (!sample.value) {
      break;
    }
    index = sample.nextIndex;
    result.push(sample.value);
  }
  return result;
}

interface CursorSample {
  value: TimestampedVector | null;
  nextIndex: number;
}

function sampleAtWithCursor(
  buffer: TimestampedVector[],
  targetNs: number,
  startIndex: number,
): CursorSample {
  let index = startIndex;
  while (index + 1 < buffer.length && buffer[index + 1].timestampNs <= targetNs) {
    index += 1;
  }

  const prev = buffer[index];
  if (!prev) {
    return { value: null, nextIndex: index };
  }

  if (prev.timestampNs === targetNs) {
    return {
      value: { x: prev.x, y: prev.y, z: prev.z, timestampNs: targetNs },
      nextIndex: index,
    };
  }

  if (prev.timestampNs < targetNs && index + 1 < buffer.length) {
    const next = buffer[index + 1];
    if (next.timestampNs > prev.timestampNs) {
      const ratio = (targetNs - prev.timestampNs) / (next.timestampNs - prev.timestampNs);
      return {
        value: {
          x: prev.x + ratio * (next.x - prev.x),
          y: prev.y + ratio * (next.y - prev.y),
          z: prev.z + ratio * (next.z - prev.z),
          timestampNs: targetNs,
        },
        nextIndex: index,
      };
    }
  }

  if (Math.abs(prev.timestampNs - targetNs) <= MAX_SYNC_SKEW_NS) {
    return {
      value: { x: prev.x, y: prev.y, z: prev.z, timestampNs: targetNs },
      nextIndex: index,
    };
  }

  if (index + 1 < buffer.length) {
    const next = buffer[index + 1];
    if (Math.abs(next.timestampNs - targetNs) <= MAX_SYNC_SKEW_NS) {
      return {
        value: { x: next.x, y: next.y, z: next.z, timestampNs: targetNs },
        nextIndex: index,
      };
    }
  }

  return { value: null, nextIndex: index };
}
