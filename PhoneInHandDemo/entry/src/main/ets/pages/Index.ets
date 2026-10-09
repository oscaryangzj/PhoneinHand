import router from '@ohos.router';
import {
  DEFAULT_SCENE_TAG,
  getHoldStateLabel,
  getSceneOption,
  RECORD_SECONDS,
  SCENE_GROUPS,
  SceneGroup,
  SceneOption,
} from '../model/SessionTypes';

@Entry
@Component
struct Index {
  @State userName: string = '';
  @State selectedSceneTag: string = DEFAULT_SCENE_TAG;

  private currentScene(): SceneOption {
    return getSceneOption(this.selectedSceneTag);
  }

  @Builder
  sceneChips(options: SceneOption[]) {
    Flex({ wrap: FlexWrap.Wrap }) {
      ForEach(options, (option: SceneOption) => {
        Text(option.tag)
          .fontSize(14)
          .fontWeight(this.selectedSceneTag === option.tag ? FontWeight.Medium : FontWeight.Regular)
          .fontColor(this.selectedSceneTag === option.tag ? Color.White : $r('app.color.text_primary'))
          .backgroundColor(this.selectedSceneTag === option.tag ? $r('app.color.primary_color') : '#E8ECF2')
          .borderRadius(14)
          .padding({ left: 14, right: 14, top: 10, bottom: 10 })
          .margin({ right: 8, bottom: 8 })
          .onClick(() => {
            this.selectedSceneTag = option.tag;
          })
      }, (option: SceneOption) => option.tag)
    }
    .width('100%')
  }

  build() {
    Column() {
      Column({ space: 4 }) {
        Text('手持检测采集')
          .fontSize(24)
          .fontWeight(FontWeight.Bold)
          .fontColor($r('app.color.text_primary'))

        Text('采集 IMU 区分手持与非手持（桌面/车内放置）')
          .fontSize(14)
          .fontColor($r('app.color.text_secondary'))
      }
      .width('100%')
      .alignItems(HorizontalAlign.Center)
      .margin({ top: 16, bottom: 8 })

      TextInput({ placeholder: $r('app.string.input_name_hint'), text: this.userName })
        .width('100%')
        .height(44)
        .fontSize(16)
        .backgroundColor(Color.White)
        .borderRadius(12)
        .padding({ left: 16, right: 16 })
        .onChange((value: string) => {
          this.userName = value.trim();
        })

      Scroll() {
        Column({ space: 10 }) {
          Text('采集场景')
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor($r('app.color.text_primary'))
            .width('100%')

          ForEach(SCENE_GROUPS, (group: SceneGroup) => {
            Column({ space: 8 }) {
              Text(`${getHoldStateLabel(group.holdState)}：`)
                .fontSize(16)
                .fontWeight(FontWeight.Bold)
                .fontColor($r('app.color.text_primary'))
                .width('100%')
              this.sceneChips(group.options)
            }
            .width('100%')
          }, (group: SceneGroup) => group.holdState)

          Text(`标签：${getHoldStateLabel(this.currentScene().holdState)} · ${this.currentScene().tag}`)
            .fontSize(13)
            .fontColor($r('app.color.primary_color'))
            .width('100%')

          Text(this.currentScene().hint)
            .fontSize(13)
            .fontColor($r('app.color.text_secondary'))
            .width('100%')

          Text(`保持当前状态采集 ${RECORD_SECONDS} 秒 · 500Hz 缓存后降采样至 100Hz`)
            .fontSize(12)
            .fontColor($r('app.color.text_secondary'))
            .width('100%')
            .margin({ top: 4 })
        }
        .width('100%')
        .justifyContent(FlexAlign.Start)
        .padding({ left: 14, right: 14, top: 16, bottom: 16 })
      }
      .width('100%')
      .layoutWeight(1)
      .scrollBar(BarState.Auto)
      .backgroundColor(Color.White)
      .borderRadius(16)
      .margin({ top: 10, bottom: 10 })

      Column({ space: 10 }) {
        Button($r('app.string.start_collection'))
          .width('100%')
          .height(48)
          .fontSize(18)
          .fontWeight(FontWeight.Medium)
          .backgroundColor($r('app.color.primary_color'))
          .borderRadius(24)
          .enabled(this.userName.length > 0)
          .opacity(this.userName.length > 0 ? 1 : 0.45)
          .onClick(() => {
            const scene = this.currentScene();
            router.pushUrl({
              url: 'pages/CollectionPage',
              params: {
                userName: this.userName,
                holdState: scene.holdState,
                sceneTag: scene.tag,
              },
            });
          })

        Button($r('app.string.realtime_demo'))
          .width('100%')
          .height(48)
          .fontSize(18)
          .fontWeight(FontWeight.Medium)
          .fontColor($r('app.color.primary_color'))
          .backgroundColor(Color.White)
          .border({ width: 1.5, color: $r('app.color.primary_color') })
          .borderRadius(24)
          .onClick(() => {
            router.pushUrl({ url: 'pages/DetectPage' });
          })
      }
      .width('100%')
    }
    .width('100%')
    .height('100%')
    .padding({ left: 20, right: 20, top: 4, bottom: 16 })
    .backgroundColor($r('app.color.page_background'))
    .alignItems(HorizontalAlign.Center)
  }
}
