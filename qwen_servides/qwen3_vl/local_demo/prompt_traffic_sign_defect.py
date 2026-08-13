TRAFFIC_SIGN_PROMPT = """
请检测图像中所有可见的交通标志牌，识别每个交通标志牌的具体类型，并检查每个交通标志牌是否存在锈蚀、遮挡、破损或外观缺陷。

你必须分别完成以下任务：

1. 检测图像中所有交通标志牌；
2. 为每个交通标志牌输出准确的边界框；
3. 判断交通标志所属的大类和具体类型；
4. 识别交通标志中的文字、数字、箭头或其他关键内容；
5. 检查每个交通标志牌是否存在指定缺陷；
6. 判断每种缺陷的大致范围和位置；
7. 只依据图像中明确可见的视觉证据作出判断，不得猜测。

────────────────────────────────────
一、交通标志类型体系
────────────────────────────────────

1. warning：警告标志
   用于警告车辆或行人注意前方危险。

   例如：
   - 急转弯
   - 连续弯路
   - 交叉路口
   - 注意行人
   - 注意儿童
   - 注意落石
   - 道路施工
   - 路面不平
   - 道路变窄
   - 注意信号灯

2. prohibition：禁令标志
   用于禁止或限制车辆、行人进行某种行为。

   例如：
   - 禁止通行
   - 禁止驶入
   - 禁止停车
   - 禁止长时间停车
   - 禁止左转
   - 禁止右转
   - 禁止掉头
   - 最高限速
   - 最低限速
   - 限高
   - 限宽
   - 限重

3. mandatory：指示标志
   用于指示车辆或行人必须按照规定方向或方式通行。

   例如：
   - 直行
   - 左转
   - 右转
   - 直行和左转
   - 直行和右转
   - 靠左行驶
   - 靠右行驶
   - 环岛行驶
   - 机动车道
   - 非机动车道
   - 步行

4. indication：道路信息指示标志
   用于指示道路设施、交通区域或通行状态。

   例如：
   - 人行横道
   - 停车场
   - 公交车站
   - 掉头位置
   - 单行路
   - 此路不通
   - 允许停车

5. guide：指路标志
   用于提供道路方向、地点、距离和出口等导航信息。

   例如：
   - 道路名称
   - 地点名称
   - 行驶方向
   - 距离信息
   - 出口信息
   - 高速公路入口或出口
   - 车道方向

6. tourist：旅游区标志
   用于指示旅游景区、旅游设施、方向或距离。

7. supplementary：辅助标志
   用于补充说明主标志的时间、范围、距离、车辆类型或其他限制条件。

8. unknown：未知类型
   当图像模糊、遮挡严重、标志内容不可辨认，或者无法可靠判断类别时使用。

────────────────────────────────────
二、缺陷类型体系
────────────────────────────────────

每个交通标志牌都必须分别检查以下四种缺陷。

1. corrosion：锈蚀

   包括：
   - 红褐色或深褐色锈斑
   - 金属氧化
   - 腐蚀痕迹
   - 边缘锈蚀
   - 螺栓、连接件或支撑结构生锈
   - 因腐蚀造成的表面粗糙、孔蚀或材料缺失

   不要误判：
   - 泥污
   - 阴影
   - 光照造成的颜色变化
   - 褪色
   - 正常的棕色图案
   - 建筑物或背景中的锈蚀

2. occlusion：遮挡

   包括：
   - 树叶或树枝遮挡
   - 车辆遮挡
   - 行人遮挡
   - 建筑物或其他设施遮挡
   - 其他交通标志遮挡
   - 广告、贴纸或其他物体覆盖
   - 标志内容或边缘被部分或完全遮住

   遮挡面积应相对于该交通标志牌完整表面的估计面积进行判断，而不是只相对于当前可见部分判断。

3. damage：破损

   包括：
   - 破裂
   - 裂缝
   - 缺角
   - 孔洞
   - 部分缺失
   - 断裂
   - 明显凹陷
   - 严重弯曲或变形
   - 标志牌与支撑结构脱离
   - 结构性破坏

   不要将轻微污渍、反光、拍摄模糊或单纯褪色判断为破损。

4. cosmetics：外观缺陷

   包括：
   - 明显褪色
   - 表面油漆剥落
   - 表面涂层脱落
   - 污渍
   - 脏污
   - 划痕
   - 表面起皮
   - 非结构性的局部外观劣化
   - 图案或文字因老化而清晰度下降

   外观缺陷不包括结构性裂缝、缺角或孔洞；这些应归入 damage。

────────────────────────────────────
三、缺陷范围等级
────────────────────────────────────

缺陷范围应根据该缺陷占交通标志牌完整表面面积的估计比例判断。

1. none
   - 未发现该缺陷；
   - 面积比例为 0；
   - detected 必须为 false。

2. tiny
   - 缺陷面积大于 0，但不超过交通标志牌面积的 2%；
   - 仅有极小的局部痕迹。

3. small
   - 缺陷面积大于 2%，但不超过 10%。

4. medium
   - 缺陷面积大于 10%，但不超过 30%。

5. large
   - 缺陷面积大于 30%；
   - 或缺陷已经明显影响标志牌主要内容、轮廓或可识别性。

面积比例使用 0 到 1 之间的小数表示，例如：

- 2% 表示为 0.02；
- 15% 表示为 0.15；
- 50% 表示为 0.50。

如果无法可靠估计面积比例，可以给出保守估计，但不得无依据夸大范围。

────────────────────────────────────
四、坐标定义
────────────────────────────────────

所有 bbox_norm_1000 均使用相对于整张输入图像宽高的 0 至 1000 归一化坐标：

[x1, y1, x2, y2]

其中：

- x1：左上角横坐标；
- y1：左上角纵坐标；
- x2：右下角横坐标；
- y2：右下角纵坐标；
- 坐标必须满足 0 <= x1 < x2 <= 1000；
- 坐标必须满足 0 <= y1 < y2 <= 1000。

交通标志牌的 bbox_norm_1000 应尽量完整覆盖标志牌主体，不要包含过多背景。

缺陷区域的 bbox_norm_1000 应尽量覆盖实际缺陷区域，而不是整个交通标志牌。

如果同一种缺陷存在多个互不相连的区域，应在 regions 中分别列出。

────────────────────────────────────
五、输出格式
────────────────────────────────────

请严格输出一个合法 JSON 对象。

不要输出 Markdown。
不要输出代码块标记。
不要在 JSON 前后添加解释文字。
不要输出注释。
不要省略规定字段。
不要使用 NaN、Infinity、None 或其他非 JSON 值。

输出格式如下：

{
  "has_traffic_sign": true,
  "objects": [
    {
      "id": 1,
      "category": "warning",
      "specific_type": "pedestrian_crossing_warning",
      "chinese_name": "注意行人",
      "visible_text": "",
      "shape": "triangle",
      "main_colors": ["red", "white", "black"],
      "bbox_norm_1000": [120, 180, 310, 460],
      "confidence": 0.95,
      "reason": "红边白底三角形，内部为行人图案",
      "has_defect": true,
      "detected_defects": ["occlusion"],
      "defects": {
        "corrosion": {
          "detected": false,
          "confidence": 0.08,
          "area_level": "none",
          "area_ratio": 0.0,
          "regions": [],
          "reason": ""
        },
        "occlusion": {
          "detected": true,
          "confidence": 0.88,
          "area_level": "small",
          "area_ratio": 0.07,
          "regions": [
            {
              "bbox_norm_1000": [240, 190, 310, 310],
              "area_ratio": 0.07,
              "description": "树叶遮挡了标志牌右上部分"
            }
          ],
          "reason": "树叶遮挡了部分标志图案和右侧边缘"
        },
        "damage": {
          "detected": false,
          "confidence": 0.10,
          "area_level": "none",
          "area_ratio": 0.0,
          "regions": [],
          "reason": ""
        },
        "cosmetics": {
          "detected": false,
          "confidence": 0.15,
          "area_level": "none",
          "area_ratio": 0.0,
          "regions": [],
          "reason": ""
        }
      }
    }
  ]
}

────────────────────────────────────
六、字段要求
────────────────────────────────────

1. id
   - 从 1 开始连续编号；
   - 每个交通标志牌使用唯一 id。

2. category
   - 只能从以下值中选择：
     warning
     prohibition
     mandatory
     indication
     guide
     tourist
     supplementary
     unknown

3. specific_type
   - 使用简短、明确的英文 snake_case 命名；
   - 例如：
     speed_limit_50
     no_parking
     no_entry
     stop
     pedestrian_crossing
     pedestrian_crossing_warning
     turn_left
     turn_right
     road_work
   - 无法可靠判断具体类型时必须填写 unknown，不得猜测。

4. chinese_name
   - 使用对应的中文交通标志名称；
   - 无法确定时填写“未知交通标志”。

5. visible_text
   - 填写标志中能够清楚识别的文字、数字、道路名称或距离信息；
   - 没有文字时填写空字符串；
   - 无法辨认时填写空字符串，不要臆测。

6. shape
   - 使用简短英文描述；
   - 优先从以下值中选择：
     triangle
     circle
     rectangle
     square
     octagon
     inverted_triangle
     irregular
     unknown

7. main_colors
   - 填写交通标志牌主体中明确可见的主要颜色；
   - 使用英文颜色名称；
   - 不要把背景颜色计入其中。

8. bbox_norm_1000
   - 表示整个交通标志牌的边界框；
   - 使用整数；
   - 必须位于 0 至 1000 范围内。

9. confidence
   - 表示对交通标志检测和类型判断的综合置信度；
   - 范围为 0 到 1；
   - 图像模糊、遮挡严重或类别不明确时应降低置信度；
   - 该数值是视觉判断置信度，不是经过校准的检测器概率。

10. reason
    - 简要说明判断交通标志类型的主要视觉依据；
    - 只描述能够从图像中直接观察到的信息。

11. defects
    - corrosion、occlusion、damage、cosmetics 四个字段必须始终全部存在；
    - 不得只输出检测到的缺陷。

12. defects.<type>.detected
    - 只能是 true 或 false；
    - 只有存在明确视觉证据时才能为 true。

13. defects.<type>.confidence
    - 表示对该缺陷是否存在的判断置信度；
    - 范围为 0 到 1；
    - detected=false 时也必须输出该字段；
    - 不确定时降低 confidence，不要猜测。

14. defects.<type>.area_level
    - 只能从以下值中选择：
      none
      tiny
      small
      medium
      large

15. defects.<type>.area_ratio
    - 表示该类缺陷总面积占交通标志牌完整表面面积的估计比例；
    - 范围为 0 到 1；
    - detected=false 时必须为 0.0；
    - 存在多个区域时，填写所有区域的总面积比例；
    - 同一类别区域重叠时不得重复计算面积。

16. defects.<type>.regions
    - detected=false 时必须为 []；
    - detected=true 时至少包含一个区域；
    - 每个区域必须包含：
      bbox_norm_1000
      area_ratio
      description

17. detected_defects
    - 只能包含 detected=true 的缺陷英文名称；
    - 名称只能从以下值中选择：
      corrosion
      occlusion
      damage
      cosmetics
    - 输出顺序固定为：
      corrosion
      occlusion
      damage
      cosmetics
    - 没有缺陷时必须为 []。

18. has_defect
    - 只要四种缺陷中任意一种 detected=true，则为 true；
    - 如果四种缺陷全部 detected=false，则为 false；
    - 必须与 detected_defects 保持一致。

────────────────────────────────────
七、一致性规则
────────────────────────────────────

必须满足以下逻辑一致性：

1. detected=false 时：
   - area_level 必须为 none；
   - area_ratio 必须为 0.0；
   - regions 必须为 []；
   - reason 应为空字符串，或简短说明为什么未发现缺陷。

2. detected=true 时：
   - area_level 不能为 none；
   - area_ratio 必须大于 0；
   - regions 至少包含一个元素；
   - reason 必须说明明确的视觉证据。

3. area_level 必须与 area_ratio 一致：
   - area_ratio = 0：none
   - 0 < area_ratio <= 0.02：tiny
   - 0.02 < area_ratio <= 0.10：small
   - 0.10 < area_ratio <= 0.30：medium
   - area_ratio > 0.30：large

4. detected_defects 必须与 defects 中的 detected 字段完全一致。

5. has_defect 必须与 detected_defects 是否为空完全一致。

6. 不要因为交通标志牌老旧就默认存在缺陷。

7. 不要把以下现象误判为缺陷：
   - 阴影
   - 反光
   - 夜间高光
   - 图像压缩噪声
   - 运动模糊
   - 低分辨率造成的细节缺失
   - 正常的标志图案
   - 背景中的物体或颜色
   - 摄像机视角造成的透视变形

8. 如果某个疑似缺陷无法与阴影、反光、污渍或背景可靠区分：
   - detected 应设为 false；
   - confidence 应降低；
   - 不得猜测。

9. 如果图中存在多个交通标志牌，必须逐个输出，不能合并为一个对象。

10. 如果交通标志牌被严重遮挡但仍可确认其存在：
    - 输出该交通标志对象；
    - category 或 specific_type 无法判断时使用 unknown；
    - 将遮挡记录为 occlusion。

11. 如果某个物体只是广告牌、商店招牌、路牌支架或普通文字牌，不属于交通管理标志，不要输出。

────────────────────────────────────
八、无交通标志时的输出
────────────────────────────────────

如果图像中没有交通标志牌，只返回：

{
  "has_traffic_sign": false,
  "objects": []
}
"""
