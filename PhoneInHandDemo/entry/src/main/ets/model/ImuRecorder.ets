import { sensor } from '@kit.SensorServiceKit';
import {
  normalizeTimestampNs,
  resampleUniform,
  TimestampedVector,
} from './ImuAligner';
import {
  CAPTURE_INTERVAL_NS,
  ImuSample,
  SAMPLE_INTERVAL_NS,
  SessionPhase,
} from './SessionTypes';

export class ImuRecorder {
  private accActive: boolean = false;
  private gyroActive: boolean = false;
  /** 采集阶段缓存的 500Hz 原始流 */
  private accRaw: TimestampedVector[] = [];
  private gyroRaw: TimestampedVector[] = [];
  /** 降采样后的 100Hz 输出 */
  private samples: ImuSample[] = [];
  private userName: string = '';
  private holdState: string = 'handheld';
  private sceneTag: string = '';

  setContext(userName: string, holdState: string, sceneTag: string): void {
    this.userName = userName;
    this.holdState = holdState;
    this.sceneTag = sceneTag;
  }

  async start(): Promise<void> {
    this.samples = [];
    this.accRaw = [];
    this.gyroRaw = [];
    await this.startGyroscope();
    await this.startAccelerometer();
  }

  stop(): void {
    this.stopAccelerometer();
    this.stopGyroscope();
    this.samples = this.downsampleToOutput();
    this.accRaw = [];
    this.gyroRaw = [];
  }

  getSamples(): ImuSample[] {
    return this.samples;
  }

  private onAccCallback(data: sensor.AccelerometerResponse): void {
    this.accRaw.push({
      x: data.x,
      y: data.y,
      z: data.z,
      timestampNs: normalizeTimestampNs(data.timestamp),
    });
  }

  private onGyroCallback(data: sensor.GyroscopeResponse): void {
    this.gyroRaw.push({
      x: data.x,
      y: data.y,
      z: data.z,
      timestampNs: normalizeTimestampNs(data.timestamp),
    });
  }

  /** 将缓存的 500Hz 原始流对齐后降采样到均匀 100Hz 网格 */
  private downsampleToOutput(): ImuSample[] {
    if (this.accRaw.length < 2 || this.gyroRaw.length < 2) {
      return [];
    }

    const startNs = Math.ceil(
      Math.max(this.accRaw[0].timestampNs, this.gyroRaw[0].timestampNs) / SAMPLE_INTERVAL_NS,
    ) * SAMPLE_INTERVAL_NS;
    const endNs = Math.min(
      this.accRaw[this.accRaw.length - 1].timestampNs,
      this.gyroRaw[this.gyroRaw.length - 1].timestampNs,
    );
    if (endNs < startNs) {
      return [];
    }

    const accSeries = resampleUniform(this.accRaw, startNs, endNs, SAMPLE_INTERVAL_NS);
    const gyroSeries = resampleUniform(this.gyroRaw, startNs, endNs, SAMPLE_INTERVAL_NS);
    const count = Math.min(accSeries.length, gyroSeries.length);
    const output: ImuSample[] = [];

    for (let i = 0; i < count; i++) {
      output.push({
        timestampNs: accSeries[i].timestampNs,
        userName: this.userName,
        holdState: this.holdState,
        sceneTag: this.sceneTag,
        phase: SessionPhase.RECORDING,
        accX: accSeries[i].x,
        accY: accSeries[i].y,
        accZ: accSeries[i].z,
        gyroX: gyroSeries[i].x,
        gyroY: gyroSeries[i].y,
        gyroZ: gyroSeries[i].z,
      });
    }
    return output;
  }

  private async startAccelerometer(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        sensor.on(sensor.SensorId.ACCELEROMETER, (data: sensor.AccelerometerResponse) => {
          this.onAccCallback(data);
        }, { interval: CAPTURE_INTERVAL_NS });
        this.accActive = true;
        resolve();
      } catch (error) {
        reject(error);
      }
    });
  }

  private async startGyroscope(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        sensor.on(sensor.SensorId.GYROSCOPE, (data: sensor.GyroscopeResponse) => {
          this.onGyroCallback(data);
        }, { interval: CAPTURE_INTERVAL_NS });
        this.gyroActive = true;
        resolve();
      } catch (error) {
        reject(error);
      }
    });
  }

  private stopAccelerometer(): void {
    if (!this.accActive) {
      return;
    }
    try {
      sensor.off(sensor.SensorId.ACCELEROMETER);
      this.accActive = false;
    } catch (error) {
      console.error('Failed to stop accelerometer:', JSON.stringify(error));
    }
  }

  private stopGyroscope(): void {
    if (!this.gyroActive) {
      return;
    }
    try {
      sensor.off(sensor.SensorId.GYROSCOPE);
      this.gyroActive = false;
    } catch (error) {
      console.error('Failed to stop gyroscope:', JSON.stringify(error));
    }
  }
}
