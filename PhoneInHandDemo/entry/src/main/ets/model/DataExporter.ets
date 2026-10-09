import { fileIo as fs } from '@kit.CoreFileKit';
import { ImuSample } from './SessionTypes';

function escapeCsv(value: string): string {
  if (value.includes(',') || value.includes('"') || value.includes('\n')) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

function formatTimestamp(timestampNs: number): string {
  return (timestampNs / 1000000).toFixed(3);
}

function buildCsvContent(samples: ImuSample[]): string {
  const header =
    'timestamp_ms,user_name,hold_state,scene_tag,phase,' +
    'acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z';
  const rows = samples.map((sample: ImuSample) => {
    return [
      formatTimestamp(sample.timestampNs),
      escapeCsv(sample.userName),
      escapeCsv(sample.holdState),
      escapeCsv(sample.sceneTag),
      sample.phase,
      sample.accX.toFixed(6),
      sample.accY.toFixed(6),
      sample.accZ.toFixed(6),
      sample.gyroX.toFixed(6),
      sample.gyroY.toFixed(6),
      sample.gyroZ.toFixed(6),
    ].join(',');
  });
  return [header, ...rows].join('\n');
}

export function saveCsvToFilesDir(filesDir: string, fileName: string, samples: ImuSample[]): string {
  const filePath = `${filesDir}/${fileName}`;
  const content = buildCsvContent(samples);
  const file = fs.openSync(filePath, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE | fs.OpenMode.TRUNC);
  fs.writeSync(file.fd, content);
  fs.closeSync(file);
  return filePath;
}
