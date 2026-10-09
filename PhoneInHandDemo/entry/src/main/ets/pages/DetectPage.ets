import router from '@ohos.router';
import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { ImuStreamCollector } from '../model/ImuStreamCollector';
import {
  HoldDetector,
  MODEL_FILE_HOLD,
} from '../model/HoldDetector';

@Entry
@Component
struct DetectPage {
  @State isListening: boolean = false;
  @State isModelReady: boolean = false;
  @State statusText: string = '正在加载模型…';
  @State labelText: string = '—';
  @State probHandheld: number = 0;
  @State probNonHandheld: number = 0;
  @State isWarmup: boolean = true;
  @State isHandheld: boolean = false;
  @State debounceEnabled: boolean = true;

  private imuCollector: ImuStreamCollector = new ImuStreamCollector();
  private holdDetector: HoldDetector = new HoldDetector();
  private uiContext: common.UIAbilityContext | null = null;

  aboutToAppear(): void {
    this.uiContext = getContext(this) as common.UIAbilityContext;
    this.loadModel();
  }

  aboutToDisappear(): void {
    this.stopListening();
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

  private async loadModel(): Promise<void> {
    try {
      if (!this.uiContext) {
        throw new Error('UI context unavailable');
      }
      const raw = await this.uiContext.resourceManager.getRawFileContent(MODEL_FILE_HOLD);
      await this.holdDetector.loadModel(raw.buffer as ArrayBuffer);
      this.isModelReady = true;
      this.statusText = '模型已就绪，点击开始检测';
    } catch (error) {
      this.isModelReady = false;
      this.statusText = '模型加载失败';
      console.error('Failed to load hold model:', JSON.stringify(error));
    }
  }

  private async startListening(): Promise<void> {
    if (!this.isModelReady || this.isListening || !this.uiContext) {
      return;
    }

    try {
      await this.requestPermissions(this.uiContext);
    } catch (error) {
      console.error('Permission request failed:', JSON.stringify(error));
    }

    this.holdDetector.setDebounceEnabled(this.debounceEnabled);
    this.holdDetector.reset();
    this.isWarmup = true;
    this.labelText = '预热中';
    this.probHandheld = 0;
    this.probNonHandheld = 0;
    this.statusText = this.debounceEnabled
      ? '检测中 · TCN · 100Hz · 非手持→手持 p≥0.95/60帧，手持→非手持 p≤0.1/35帧'
      : '检测中 · TCN · 100Hz · 预热 60 帧，之后逐帧输出';

    this.holdDetector.setCallback((result) => {
      this.isWarmup = result.warmup;
      this.labelText = result.labelName;
      this.probHandheld = result.probabilityHandheld;
      this.probNonHandheld = result.probabilityNonHandheld;
      this.isHandheld = result.isHandheld;
    });

    this.imuCollector.setCallback((sample) => {
      this.holdDetector.processSample(sample);
    });

    try {
      await this.imuCollector.start();
      this.isListening = true;
    } catch (error) {
      console.error('Failed to start IMU stream:', JSON.stringify(error));
      this.statusText = 'IMU 启动失败';
    }
  }

  private stopListening(): void {
    if (!this.isListening) {
      return;
    }
    this.imuCollector.stop();
    this.isListening = false;
    this.statusText = this.isModelReady ? '已停止，可再次开始' : '模型未就绪';
  }

  private labelColor(): string {
    if (this.isWarmup || !this.isListening) {
      return '#667085';
    }
    return this.isHandheld ? '#12B76A' : '#F79009';
  }

  private circleFill(): string {
    if (this.isWarmup || !this.isListening) {
      return '#EEF2F6';
    }
    return this.isHandheld ? '#E8F8EF' : '#FFFAEB';
  }

  private formatProb(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }

  build() {
    Column() {
      Row() {
        Text('‹ 返回')
          .fontSize(16)
          .fontColor($r('app.color.primary_color'))
          .onClick(() => {
            this.stopListening();
            router.back();
          })
      }
      .width('100%')
      .margin({ top: 12, bottom: 8 })

      Text('实时手持检测')
        .fontSize(26)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .margin({ bottom: 6 })

      Text(this.statusText)
        .fontSize(14)
        .fontColor($r('app.color.text_secondary'))
        .margin({ bottom: 28 })

      Row() {
        Text('输出防抖')
          .fontSize(15)
          .fontColor($r('app.color.text_primary'))

        Toggle({ type: ToggleType.Switch, isOn: this.debounceEnabled })
          .enabled(!this.isListening)
          .onChange((isOn: boolean) => {
            this.debounceEnabled = isOn;
          })
      }
      .width('100%')
      .justifyContent(FlexAlign.SpaceBetween)
      .margin({ bottom: 20 })

      Stack({ alignContent: Alignment.Center }) {
        Circle()
          .width(220)
          .height(220)
          .fill(this.circleFill())

        Column({ space: 6 }) {
          Text(this.labelText)
            .fontSize(36)
            .fontWeight(FontWeight.Bold)
            .fontColor(this.labelColor())

          if (!this.isWarmup && this.isListening) {
            Text(`手持原始概率 ${this.formatProb(this.probHandheld)}`)
              .fontSize(14)
              .fontColor($r('app.color.text_secondary'))
          }
        }
      }
      .margin({ bottom: 28 })

      Column({ space: 10 }) {
        Text(`非手持原始概率：${this.formatProb(this.probNonHandheld)}`)
          .fontSize(15)
          .fontColor($r('app.color.text_primary'))
          .width('100%')

        Text(`手持原始概率：${this.formatProb(this.probHandheld)}`)
          .fontSize(15)
          .fontColor($r('app.color.text_primary'))
          .width('100%')

        Text(`模型：${MODEL_FILE_HOLD} · TCN 流式推理`)
          .fontSize(12)
          .fontColor($r('app.color.text_secondary'))
          .width('100%')
          .margin({ top: 4 })
      }
      .width('100%')
      .padding(16)
      .backgroundColor(Color.White)
      .borderRadius(12)

      Blank()

      Button(this.isListening ? '停止检测' : '开始检测')
        .width('100%')
        .height(52)
        .fontSize(18)
        .fontWeight(FontWeight.Medium)
        .backgroundColor(this.isListening ? '#D92D20' : $r('app.color.primary_color'))
        .borderRadius(26)
        .enabled(this.isModelReady)
        .opacity(this.isModelReady ? 1 : 0.45)
        .margin({ bottom: 24 })
        .onClick(() => {
          if (this.isListening) {
            this.stopListening();
          } else {
            this.startListening();
          }
        })
    }
    .width('100%')
    .height('100%')
    .padding({ left: 20, right: 20, top: 4, bottom: 12 })
    .backgroundColor($r('app.color.page_background'))
    .alignItems(HorizontalAlign.Center)
  }
}
