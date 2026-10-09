import router from '@ohos.router';
import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { CountdownRingView } from '../components/CountdownRingView';
import { ImuRecorder } from '../model/ImuRecorder';
import { saveCsvToFilesDir } from '../model/DataExporter';
import {
  DEFAULT_HOLD_STATE,
  DEFAULT_SCENE_TAG,
  getHoldStateLabel,
  getSceneOption,
  HoldState,
  parseHoldState,
  RECORD_SECONDS,
} from '../model/SessionTypes';

const TICK_MS: number = 50;

@Entry
@Component
struct CollectionPage {
  @State countdownValue: number = RECORD_SECONDS;
  @State remainingRatio: number = 1;
  @State displayElapsedSeconds: number = 0;
  @State statusHint: string = '请准备';
  @State sceneTag: string = DEFAULT_SCENE_TAG;
  @State holdStateLabel: string = '手持';
  @State preparing: boolean = true;
  private userName: string = '';
  private holdState: HoldState = DEFAULT_HOLD_STATE;
  private recorder: ImuRecorder = new ImuRecorder();
  private segmentTimer: number = -1;
  private sessionStarted: boolean = false;
  private remainingMs: number = RECORD_SECONDS * 1000;

  aboutToAppear(): void {
    const params = router.getParams() as Record<string, Object>;
    this.userName = (params?.userName as string) ?? '';
    this.holdState = parseHoldState(params?.holdState as string);
    this.holdStateLabel = getHoldStateLabel(this.holdState);
    this.sceneTag = (params?.sceneTag as string) ?? DEFAULT_SCENE_TAG;
    if (this.userName.length === 0) {
      router.replaceUrl({ url: 'pages/Index' });
      return;
    }
    this.recorder.setContext(this.userName, this.holdState, this.sceneTag);
    this.startSession();
  }

  aboutToDisappear(): void {
    this.clearTimer();
    this.recorder.stop();
  }

  private clearTimer(): void {
    if (this.segmentTimer >= 0) {
      clearInterval(this.segmentTimer);
      this.segmentTimer = -1;
    }
  }

  private async requestPermissions(context: common.UIAbilityContext): Promise<boolean> {
    const permissions: Permissions[] = [
      'ohos.permission.ACCELEROMETER',
      'ohos.permission.GYROSCOPE',
    ];
    const atManager = abilityAccessCtrl.createAtManager();
    const result = await atManager.requestPermissionsFromUser(context, permissions);
    return result.authResults.every(
      (status: abilityAccessCtrl.GrantStatus) => status === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED,
    );
  }

  private async startSession(): Promise<void> {
    if (this.sessionStarted) {
      return;
    }
    this.sessionStarted = true;
    this.preparing = true;
    this.statusHint = '正在启动 IMU 传感器…';

    const context = getContext(this) as common.UIAbilityContext;
    try {
      await this.requestPermissions(context);
    } catch (error) {
      console.error('Permission request failed:', JSON.stringify(error));
    }

    try {
      await this.recorder.start();
      this.preparing = false;
      this.beginRecording();
    } catch (error) {
      console.error('Failed to start IMU sensors:', JSON.stringify(error));
      this.preparing = false;
      this.statusHint = 'IMU 启动失败，请重试';
      this.sessionStarted = false;
      this.finishSession();
    }
  }

  private beginRecording(): void {
    this.remainingMs = RECORD_SECONDS * 1000;
    this.countdownValue = RECORD_SECONDS;
    this.remainingRatio = 1;
    this.displayElapsedSeconds = 0;
    this.statusHint = getSceneOption(this.sceneTag).hint;

    this.runTimer(RECORD_SECONDS * 1000, () => {
      this.finishSession();
    });
  }

  private runTimer(totalMs: number, onFinished: () => void): void {
    this.clearTimer();
    this.remainingMs = totalMs;
    this.segmentTimer = setInterval(() => {
      this.remainingMs -= TICK_MS;
      if (this.remainingMs <= 0) {
        this.clearTimer();
        this.remainingMs = 0;
        this.remainingRatio = 0;
        this.countdownValue = 0;
        this.displayElapsedSeconds = RECORD_SECONDS;
        onFinished();
        return;
      }
      this.remainingRatio = this.remainingMs / totalMs;
      this.countdownValue = Math.max(1, Math.ceil(this.remainingMs / 1000));
      const elapsed = Math.floor((RECORD_SECONDS * 1000 - this.remainingMs) / 1000);
      if (elapsed !== this.displayElapsedSeconds) {
        this.displayElapsedSeconds = elapsed;
      }
    }, TICK_MS);
  }

  private finishSession(): void {
    this.clearTimer();
    this.recorder.stop();

    const context = getContext(this) as common.UIAbilityContext;
    const timestamp = new Date();
    const fileName =
      `imu_inhand_${this.userName}_${this.holdState}_${this.sceneTag}_` +
      `${timestamp.getFullYear()}${this.pad2(timestamp.getMonth() + 1)}${this.pad2(timestamp.getDate())}_` +
      `${this.pad2(timestamp.getHours())}${this.pad2(timestamp.getMinutes())}` +
      `${this.pad2(timestamp.getSeconds())}.csv`;
    const filePath = saveCsvToFilesDir(context.filesDir, fileName, this.recorder.getSamples());

    router.replaceUrl({
      url: 'pages/ResultPage',
      params: {
        userName: this.userName,
        holdStateLabel: this.holdStateLabel,
        sceneTag: this.sceneTag,
        sampleCount: this.recorder.getSamples().length,
        filePath,
      },
    });
  }

  private pad2(value: number): string {
    return value < 10 ? `0${value}` : `${value}`;
  }

  build() {
    Column() {
      Text(`${this.userName} · ${this.holdStateLabel} · ${this.sceneTag}`)
        .fontSize(15)
        .fontColor($r('app.color.text_secondary'))
        .margin({ top: 28, bottom: 6 })

      if (this.preparing) {
        Text('IMU 启动中')
          .fontSize(24)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.primary_color'))
          .margin({ bottom: 6 })

        Text('即将开始采集')
          .fontSize(16)
          .fontColor($r('app.color.text_secondary'))
          .margin({ bottom: 12 })
      } else {
        Text(this.sceneTag)
          .fontSize(26)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.primary_color'))
          .margin({ bottom: 10 })

        Text(this.holdStateLabel)
          .fontSize(16)
          .fontColor($r('app.color.recording_color'))
          .margin({ bottom: 10 })
      }

      CountdownRingView({
        remainingRatio: this.remainingRatio,
        countdownValue: Math.max(1, this.countdownValue),
      })

      Text(this.statusHint)
        .fontSize(14)
        .fontColor($r('app.color.text_secondary'))
        .margin({ top: 16 })
        .padding({ left: 24, right: 24 })
        .textAlign(TextAlign.Center)

      if (!this.preparing) {
        Progress({
          value: this.displayElapsedSeconds,
          total: RECORD_SECONDS,
          type: ProgressType.Linear,
        })
          .width('84%')
          .margin({ top: 16 })

        Text(`采集进度 ${this.displayElapsedSeconds}/${RECORD_SECONDS} 秒`)
          .fontSize(13)
          .fontColor($r('app.color.text_secondary'))
          .margin({ top: 8 })
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.page_background'))
    .alignItems(HorizontalAlign.Center)
  }
}
