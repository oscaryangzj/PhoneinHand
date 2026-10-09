import { mindSporeLite } from '@kit.MindSporeLiteKit';
import { StreamImuSample } from './ImuStreamCollector';

/** R10: R9 plus two both-hands-static training sessions; best epoch 12/50; same normalization and FSM. */
export const MODEL_FILE_HOLD: string = 'phone_inhand_step_r10_r9_both_static2_smart2_other1_rot3_tmse_50epochs_20261008.ms';
const INPUT_DIM: number = 6;
const NUM_CLASSES: number = 2;
const STEP_CHUNK_SIZE: number = 1;
const CNN_CHANNELS: number = 32;
const CNN_STATE_LEN: number = 60;
const ACC_CHANNELS: number = 3;
const RECEPTIVE_FIELD: number = 61;
const HANDHELD_THRESHOLD: number = 0.5;
const HANDHELD_EVIDENCE_THRESHOLD: number = 0.95;
const NON_HANDHELD_EVIDENCE_THRESHOLD: number = 0.1;
const STABLE_EVIDENCE_RATIO: number = 0.9;
const CONTACT_TO_FREE_WINDOW_FRAMES: number = 60;
const FREE_TO_CONTACT_WINDOW_FRAMES: number = 35;
const EVIDENCE_BUFFER_FRAMES: number = Math.max(CONTACT_TO_FREE_WINDOW_FRAMES, FREE_TO_CONTACT_WINDOW_FRAMES);

/** Fixed train-set norm_stats (acc_x/y/z, gyro_x/y/z). Shared by R3, R5a, R6b, R7, R8, R9 and R10. */
const NORM_MEAN: number[] = [
  0.05910849943757057,
  0.4817154109477997,
  9.445425987243652,
  0.00010093271703226492,
  -0.00027028663316741586,
  -0.0015120282769203186,
];
const NORM_STD: number[] = [
  1.617651104927063,
  1.9742977619171143,
  0.6221764087677002,
  0.0968429297208786,
  0.11744307726621628,
  0.314462274312973,
];

export const LABEL_NON_HANDHELD: number = 0;
export const LABEL_HANDHELD: number = 1;
export const LABEL_DISPLAY_NAMES: string[] = ['非手持', '手持'];
export type HoldDecisionState = 'unknown' | 'non_handheld' | 'handheld';
export type HoldResultStatus = 'warming' | 'settling' | 'ready';

export interface HoldResult {
  state: HoldDecisionState;
  status: HoldResultStatus;
  labelIndex: number;
  labelName: string;
  probabilityHandheld: number;
  probabilityNonHandheld: number;
  isHandheld: boolean;
  warmup: boolean;
}

export type HoldCallback = (result: HoldResult) => void;

/**
 * 流式 Causal CNN：每帧 imu[1,1,6] + cnn_buffer[1,32,60] + grav_state[1,3]
 * → logits + state + grav_state_out（模型内 EMA 去重力）。
 */
export class HoldDetector {
  private model: mindSporeLite.Model | null = null;
  private cnnBuffer: Float32Array = new Float32Array(CNN_CHANNELS * CNN_STATE_LEN);
  private gravState: Float32Array = new Float32Array(ACC_CHANNELS);
  private gravInited: boolean = false;
  private imuChunk: Float32Array = new Float32Array(STEP_CHUNK_SIZE * INPUT_DIM);
  private evidence: number[] = [];
  private decisionFrameCount: number = 0;
  private currentLabel: number = -1;
  private frameIndex: number = 0;
  private processing: boolean = false;
  private sampleQueue: StreamImuSample[] = [];
  private onResult: HoldCallback | null = null;
  private debounceEnabled: boolean = true;

  async loadModel(buffer: ArrayBuffer): Promise<void> {
    const context: mindSporeLite.Context = {
      target: ['cpu'],
      cpu: {
        threadNum: 1,
        threadAffinityMode: 1,
        precisionMode: 'enforce_fp32',
      },
    };
    this.model = await mindSporeLite.loadModelFromBuffer(buffer, context);
  }

  setCallback(callback: HoldCallback): void {
    this.onResult = callback;
  }

  setDebounceEnabled(enabled: boolean): void {
    this.debounceEnabled = enabled;
    this.evidence = [];
    this.decisionFrameCount = 0;
    this.currentLabel = -1;
  }

  reset(): void {
    this.cnnBuffer.fill(0);
    this.gravState.fill(0);
    this.gravInited = false;
    this.imuChunk.fill(0);
    this.evidence = [];
    this.decisionFrameCount = 0;
    this.currentLabel = -1;
    this.frameIndex = 0;
    this.processing = false;
    this.sampleQueue = [];
  }

  async processSample(sample: StreamImuSample): Promise<void> {
    if (this.model === null) {
      return;
    }
    this.enqueueSample(sample);
    if (this.processing) {
      return;
    }
    this.processing = true;
    try {
      while (this.sampleQueue.length > 0) {
        const next = this.sampleQueue.shift();
        if (next === undefined) {
          break;
        }
        await this.runSample(next);
      }
    } finally {
      this.processing = false;
    }
  }

  private enqueueSample(sample: StreamImuSample): void {
    this.sampleQueue.push(sample);
  }

  private async runSample(sample: StreamImuSample): Promise<void> {
    const normalized = this.normalizeSample(sample);
    for (let i = 0; i < INPUT_DIM; i++) {
      this.imuChunk[i] = normalized[i];
    }
    // 首帧用当前加速度初始化重力状态，与训练时 None→首帧 行为一致
    if (!this.gravInited) {
      this.gravState[0] = normalized[0];
      this.gravState[1] = normalized[1];
      this.gravState[2] = normalized[2];
      this.gravInited = true;
    }

    try {
      const logits = await this.runStepInference();
      // logits layout [1, T, C] with T=1 → [non_handheld, handheld]
      const probs = this.softmax2(logits[0], logits[1]);
      const pHandheld = probs[1];

      const modelWarmup = this.frameIndex < (RECEPTIVE_FIELD - 1);
      if (!modelWarmup) {
        if (this.debounceEnabled) {
          this.stabilize(pHandheld);
        } else {
          this.currentLabel = pHandheld >= HANDHELD_THRESHOLD ? LABEL_HANDHELD : LABEL_NON_HANDHELD;
        }
      }
      this.frameIndex += STEP_CHUNK_SIZE;
      const unresolved = modelWarmup || (this.debounceEnabled && this.currentLabel < 0);
      const warmup = modelWarmup || unresolved;
      const state: HoldDecisionState = unresolved ? 'unknown' :
        this.currentLabel === LABEL_HANDHELD ? 'handheld' : 'non_handheld';
      const status: HoldResultStatus = modelWarmup ? 'warming' :
        this.debounceEnabled && unresolved ? 'settling' : 'ready';
      if (this.onResult) {
        this.onResult({
          state: state,
          status: status,
          labelIndex: unresolved ? -1 : this.currentLabel,
          labelName: modelWarmup ? '预热中' : unresolved ? '判定中' : LABEL_DISPLAY_NAMES[this.currentLabel],
          probabilityHandheld: pHandheld,
          probabilityNonHandheld: probs[0],
          isHandheld: !unresolved && this.currentLabel === LABEL_HANDHELD,
          warmup: warmup,
        });
      }
    } catch (error) {
      console.error('HoldDetector step inference failed:', JSON.stringify(error));
    }
  }

  private stabilize(probabilityHandheld: number): void {
    let candidate = -1;
    if (probabilityHandheld >= HANDHELD_EVIDENCE_THRESHOLD) {
      candidate = LABEL_HANDHELD;
    } else if (probabilityHandheld <= NON_HANDHELD_EVIDENCE_THRESHOLD) {
      candidate = LABEL_NON_HANDHELD;
    }

    this.decisionFrameCount += 1;
    this.evidence.push(candidate);
    if (this.evidence.length > EVIDENCE_BUFFER_FRAMES) {
      this.evidence.shift();
    }

    let target = -1;
    let windowFrames = 0;
    if (this.currentLabel < 0) {
      if (candidate === LABEL_NON_HANDHELD) {
        target = LABEL_NON_HANDHELD;
        windowFrames = FREE_TO_CONTACT_WINDOW_FRAMES;
      } else if (candidate === LABEL_HANDHELD) {
        target = LABEL_HANDHELD;
        windowFrames = CONTACT_TO_FREE_WINDOW_FRAMES;
      }
    } else if (this.currentLabel === LABEL_NON_HANDHELD && candidate === LABEL_HANDHELD) {
      target = LABEL_HANDHELD;
      windowFrames = CONTACT_TO_FREE_WINDOW_FRAMES;
    } else if (this.currentLabel === LABEL_HANDHELD && candidate === LABEL_NON_HANDHELD) {
      target = LABEL_NON_HANDHELD;
      windowFrames = FREE_TO_CONTACT_WINDOW_FRAMES;
    }

    if (target >= 0 && this.hasEvidence(target, windowFrames)) {
      this.currentLabel = target;
      this.evidence = [];
      this.decisionFrameCount = 0;
    }
  }

  private hasEvidence(target: number, windowFrames: number): boolean {
    if (this.decisionFrameCount < windowFrames || this.evidence.length < windowFrames) {
      return false;
    }
    let supportingSamples = 0;
    let latestLabel = -1;
    const start = this.evidence.length - windowFrames;
    for (let i = start; i < this.evidence.length; i++) {
      const label = this.evidence[i];
      latestLabel = label;
      if (label === target) {
        supportingSamples += 1;
      }
    }

    const requiredSamples = Math.ceil(windowFrames * STABLE_EVIDENCE_RATIO);
    return latestLabel === target && supportingSamples >= requiredSamples;
  }

  private normalizeSample(sample: StreamImuSample): Float32Array {
    const raw = [sample.accX, sample.accY, sample.accZ, sample.gyroX, sample.gyroY, sample.gyroZ];
    const out = new Float32Array(INPUT_DIM);
    for (let c = 0; c < INPUT_DIM; c++) {
      const std = NORM_STD[c] > 1e-6 ? NORM_STD[c] : 1e-6;
      out[c] = (raw[c] - NORM_MEAN[c]) / std;
    }
    return out;
  }

  private async runStepInference(): Promise<Float32Array> {
    if (this.model === null) {
      return new Float32Array(NUM_CLASSES);
    }
    const inputs = this.model.getInputs();
    inputs[0].setData(this.imuChunk.buffer);
    inputs[1].setData(this.cnnBuffer.buffer);
    inputs[2].setData(this.gravState.buffer);
    const outputs = await this.model.predict(inputs);
    // outputs: logits, cnn_buffer_out, grav_state_out
    const logits = new Float32Array(outputs[0].getData());
    this.cnnBuffer = new Float32Array(outputs[1].getData());
    this.gravState = new Float32Array(outputs[2].getData());
    return logits;
  }

  private softmax2(logit0: number, logit1: number): number[] {
    const maxL = Math.max(logit0, logit1);
    const e0 = Math.exp(logit0 - maxL);
    const e1 = Math.exp(logit1 - maxL);
    const sum = e0 + e1;
    return [e0 / sum, e1 / sum];
  }

}
