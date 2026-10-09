/**
 * 手持状态：手持 / 非手持（桌面、车载支架、座椅等放置）
 */
export enum HoldState {
  HANDHELD = 'handheld',
  NON_HANDHELD = 'non_handheld',
}

export enum SessionPhase {
  RECORDING = 'recording',
}

/** 单次采集时长（秒） */
export const RECORD_SECONDS: number = 15;

/** 场景标签；同时决定 hold_state */
export interface SceneOption {
  tag: string;
  holdState: HoldState;
  hint: string;
}

export interface SceneGroup {
  holdState: HoldState;
  options: SceneOption[];
}

export const SCENE_OPTIONS: SceneOption[] = [
  {
    tag: '手持静止（左手）',
    holdState: HoldState.HANDHELD,
    hint: '左手单手握持，手机全程完全离开桌面，自然保持静止',
  },
  {
    tag: '手持静止（右手）',
    holdState: HoldState.HANDHELD,
    hint: '右手单手握持，手机全程完全离开桌面，自然保持静止',
  },
  {
    tag: '手持静止（双手）',
    holdState: HoldState.HANDHELD,
    hint: '双手握持，手机全程完全离开桌面，自然保持静止',
  },
  {
    tag: '手持微动',
    holdState: HoldState.HANDHELD,
    hint: '手握手机，完全离开桌面后轻轻摇动，幅度尽量与桌面微动相近',
  },
  { tag: '手持步行', holdState: HoldState.HANDHELD, hint: '手持手机自然步行，手机全程完全离开桌面' },
  { tag: '桌面静置', holdState: HoldState.NON_HANDHELD, hint: '手机平放桌面，勿触碰' },
  {
    tag: '桌面微动',
    holdState: HoldState.NON_HANDHELD,
    hint: '手握手机，保持接触点基本不滑动，围绕接触点轻轻摇动，全程不离桌',
  },
  { tag: '车载支架', holdState: HoldState.NON_HANDHELD, hint: '固定在车载支架上，可静止或行驶' },
  {
    tag: '车载手持',
    holdState: HoldState.HANDHELD,
    hint: '由乘客手握手机，全程不接触座椅、中控台或支架等支撑物；车辆可静止或行驶',
  },
  {
    tag: '车载静置',
    holdState: HoldState.NON_HANDHELD,
    hint: '手机放在座椅或中控台等车内位置，全程不拿起；车辆可静止或行驶，支架场景请选择车载支架',
  },
];

export const SCENE_GROUPS: SceneGroup[] = [
  {
    holdState: HoldState.HANDHELD,
    options: SCENE_OPTIONS.filter((option: SceneOption) => option.holdState === HoldState.HANDHELD),
  },
  {
    holdState: HoldState.NON_HANDHELD,
    options: SCENE_OPTIONS.filter((option: SceneOption) => option.holdState === HoldState.NON_HANDHELD),
  },
];

export const DEFAULT_SCENE_TAG: string = SCENE_OPTIONS[0].tag;
export const DEFAULT_HOLD_STATE: HoldState = SCENE_OPTIONS[0].holdState;

export function getHoldStateLabel(state: HoldState): string {
  return state === HoldState.HANDHELD ? '手持' : '非手持';
}

export function getSceneOption(tag: string): SceneOption {
  for (let i = 0; i < SCENE_OPTIONS.length; i++) {
    if (SCENE_OPTIONS[i].tag === tag) {
      return SCENE_OPTIONS[i];
    }
  }
  return SCENE_OPTIONS[0];
}

export function parseHoldState(value: string | undefined): HoldState {
  if (value === HoldState.NON_HANDHELD || value === 'non_handheld') {
    return HoldState.NON_HANDHELD;
  }
  return HoldState.HANDHELD;
}

/** 传感器采集缓存频率 */
export const CAPTURE_RATE_HZ: number = 500;
export const CAPTURE_INTERVAL_NS: number = 1000000000 / CAPTURE_RATE_HZ;
/** 落盘输出频率（由 500Hz 降采样得到） */
export const SAMPLE_RATE_HZ: number = 100;
export const SAMPLE_INTERVAL_MS: number = 1000 / SAMPLE_RATE_HZ;
export const SAMPLE_INTERVAL_NS: number = SAMPLE_INTERVAL_MS * 1000000;

export interface ImuSample {
  timestampNs: number;
  userName: string;
  holdState: string;
  sceneTag: string;
  phase: SessionPhase;
  accX: number;
  accY: number;
  accZ: number;
  gyroX: number;
  gyroY: number;
  gyroZ: number;
}
