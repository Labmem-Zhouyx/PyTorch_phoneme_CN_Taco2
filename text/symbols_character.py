'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
_pad        = '_'
_eos        = '~'
_puncts     = '!\'\"(),-.:;? %/'
_characters = ['拣', '撰', '普', '黄', '跳', '棱', '润', '扰', '绮', '牺', '鲁', '岚', '弓', '惧', '叠', '嬅', '手', '憔', '是', '薯', '揶', '烁', '姚', '侵', \
           '蔺', '杰', '杏', '侃', '跶', '失', '晔', '塑', '苹', '枚', '悼', '焜', '内', '纽', '邵', '宿', '盖', '胰', '想', '魂', '己', '奇', '库', '揣', \
           '讳', '鸵', '洪', '鳕', '扮', '评', '士', '焚', '文', '挖', '若', '啼', '癌', '娄', '攻', '啷', '恩', '奉', '穹', '旗', '彦', '菠', '挫', '侈', \
           '亚', '飘', '螳', '吓', '缀', '凳', '器', '讶', '体', '妾', '峇', '突', '仄', '材', '滧', '嘈', '娟', '趋', '垟', '曼', '礴', '樊', '栅', '姨', \
           '阵', '顽', '焰', '圣', '就', '豪', '严', '听', '淳', '零', '圻', '虹', '改', '故', '范', '银', '伏', '细', '养', '晏', '霞', '筛', '蹲', '栩', '喇', '狼', '点', '逆', '浪', '鹰', '季', '怡', '结', '情', '澜', '列', '悠', '萌', '迫', '诬', '桑', '氰', '唬', '尽', '闯', '弦', '役', '睁', '愧', '炉', '踉', '张', '义', '舫', '踝', '毅', '卤', '悔', '冗', '柔', '船', '恼', '疵', '匈', '舟', '滦', '僻', '抄', '栋', '璨', '龌', '耐', '居', '吐', '祈', '缠', '毽', '抢', '拙', '寔', '典', '逞', '盈', '宠', '辱', '宁', '藜', '妩', '耶', '岳', '潢', '等', '础', '叮', '铬', '楷', '罪', '轴', '摆', '佑', '壤', '包', '卯', '正', '呲', '铛', '纪', '誊', '洒', '修', '巫', '答', '抒', '斟', '谌', '牲', '偕', '愿', '观', '融', '侧', '膀', '飨', '习', '饺', '深', '支', '汪', '截', '悲', '湮', '馈', '烤', '婉', '望', '创', '韦', '珐', '奈', '碘', '饼', '逮', '柱', '异', '姑', '厌', '律', '按', '攒', '炯', '便', '莠', '酒', '峙', '妹', '钉', '麦', '脆', '真', '规', '播', '毡', '亲', '菇', '玺', '心', '倩', '腰', '锣', '爸', '犊', '危', '殊', '击', '乡', '鳍', '炙', '苛', '极', '鸣', '戳', '旦', '邢', '救', '露', '眠', '较', '茎', '兔', '銮', '瞬', '啰', '苏', '喵', '人', '径', '务', '雅', '钾', '皙', '脱', '天', '号', '矣', '葫', '驮', '旮', '塘', '渐', '硕', '惕', '磬', '楠', '桃', '秉', '前', '礤', '庹', '沪', '隶', '尊', '互', '迈', '邋', '芒', '厥', '纷', '呦', '孪', '簸', '蚝', '谢', '综', '秸', '秆', '沙', '延', '榴', '帕', '丈', '叔', '狄', '并', '都', '蹚', '足', '唷', '焦', '权', '竹', '瘘', '咦', '车', '勺', '冻', '整', '萃', '脾', '綮', '丝', '蕾', '疫', '肪', '吵', '催', '窘', '硬', '灭', '醒', '柳', '既', '蒙', '嫖', '泽', '鞭', '毗', '敲', '逾', '邦', '蒲', '洣', '岐', '拔', '捺', '审', '昭', '秃', '苇', '途', '家', '找', '撑', '罩', '亨', '峪', '祭', '忘', '账', '留', '裢', '馄', '暇', '茏', '另', '醛', '甄', '沤', '抿', '竣', '韭', '痴', '件', '潜', '论', '盾', '柯', '雹', '我', '荔', '磨', '巴', '雾', '捏', '辉', '刺', '塌', '臧', '湍', '倒', '馊', '监', '浣', '拖', '宽', '炮', '殷', '谭', '旁', '邓', '銶', '狙', '踪', '浚', '误', '寒', '藉', '唾', '籍', '认', '峭', '嚓', '灞', '仗', '慷', '祾', '蟀', '拱', '疑', '劳', '卒', '唔', '嗝', '避', '君', '眶', '郎', '拄', '胜', '腻', '泮', '爽', '则', '颤', '纭', '褥', '豫', '叟', '怯', '事', '仪', '凸', '忧', '表', '敛', '抚', '趄', '猴', '铮', '冯', '噎', '胎', '泳', '透', '伐', '呖', '瞧', '扇', '边', '芍', '垫', '漂', '窝', '挤', '廊', '粗', '坂', '舔', '瘩', '噘', '崖', '桌', '奔', '昌', '殴', '斑', '姥', '双', '痕', '觑', '惠', '滨', '扛', '健', '徘', '窟', '采', '各', '辍', '斤', '年', '撵', '瞟', '鄞', '娜', '棚', '逯', '缸', '从', '嗯', '知', '骏', '剌', '鼻', '访', '水', '册', '脂', '拇', '吸', '赤', '斯', '瞪', '灵', '奏', '俏', '叨', '蜘', '拦', '姈', '顷', '货', '掉', '去', '傀', '橡', '篇', '庚', '寥', '梁', '糕', '解', '记', '澳', '襄', '裳', '脏', '趁', '陕', '街', '流', '胺', '校', '涣', '荡', '虎', '俊', '凛', '但', '信', '酶', '硝', '癍', '虱', '篆', '螺', '装', '制', '屎', '膊', '忆', '览', '凑', '呈', '扬', '昼', '牵', '郝', '插', '秘', '瓶', '鹧', '膜', '矛', '康', '苍', '厮', '耻', '孖', '聊', '雷', '友', '驼', '甲', '盒', '汽', '晌', '纵', '袭', '疯', '挪', '孻', '凌', '嗪', '撞', '烹', '霾', '罕', '案', '倪', '纫', '棍', '你', '薄', '白', '魅', '挽', '官', '总', '达', '琴', '揭', '济', '野', '良', '别', '李', '缥', '楚', '烂', '步', '伤', '古', '乾', '姜', '肮', '俞', '驴', '鄱', '绳', '其', '萄', '史', '舆', '崽', '悴', '冉', '清', '晒', '锤', '使', '诸', '酵', '欢', '通', '泛', '力', '鹈', '琅', '呃', '类', '际', '诊', '辗', '夜', '组', '睫', '咽', '朴', '湿', '匮', '邱', '绥', '屈', '佩', '纾', '逛', '矬', '卓', '举', '颚', '履', '昕', '炽', '守', '方', '怪', '妈', '痔', '汛', '垓', '允', '萎', '漱', '摩', '牌', '炎', '多', '封', '滉', '袒', '洁', '准', '肆', '螭', '愤', '剪', '丐', '央', '珲', '扯', '滥', '激', '傲', '奎', '匙', '闲', '憋', '明', '邻', '猖', '券', '畏', '熊', '午', '霓', '脚', '衰', '测', '弹', '已', '岁', '种', '腮', '躯', '缆', '牙', '辐', '迄', '克', '漆', '积', '罹', '层', '巡', '暴', '犀', '葆', '身', '备', '孽', '且', '晚', '禾', '媛', '息', '敏', '颊', '柄', '掏', '怆', '洲', '辣', '沁', '梗', '盯', '雨', '哄', '有', '金', '肺', '恃', '晃', '谨', '膨', '索', '滕', '铵', '招', '镶', '夙', '裔', '宰', '轰', '才', '趣', '朵', '棕', '铐', '鬃', '摽', '璧', '富', '众', '早', '岩', '慰', '恢', '宪', '视', '辻', '穴', '浊', '姊', '锚', '林', '脑', '凉', '硼', '粉', '遍', '胞', '位', '剥', '短', '虡', '航', '神', '顼', '腹', '昀', '馒', '扔', '榆', '苦', '坊', '般', '锦', '武', '幽', '南', '族', '澡', '垮', '简', '番', '吠', '顺', '啸', '纰', '舛', '氧', '涝', '凭', '软', '洄', '叫', '桠', '单', '算', '药', '钥', '漫', '翱', '蹄', '蛮', '涛', '概', '咨', '择', '骸', '敦', '潘', '否', '镑', '虑', '右', '海', '娃', '绰', '址', '瞠', '瑄', '寻', '癣', '疡', '捆', '晦', '树', '佐', '湟', '菩', '钞', '蔬', '摸', '涟', '瑜', '揄', '吭', '皖', '用', '淤', '印', '舱', '赈', '鬼', '衙', '效', '弄', '您', '丫', '鱼', '暂', '玄', '秋', '曾', '钻', '擒', '纬', '郁', '音', '瓢', '局', '胃', '凿', '遏', '翅', '群', '瘙', '噻', '潲', '恍', '尺', '缇', '髓', '猾', '盆', '腥', '贯', '哎', '翚', '钢', '嵩', '爬', '拭', '裘', '濡', '唐', '哭', '俯', '噌', '乓', '左', '惚', '上', '翻', '辅', '篱', '迟', '秤', '非', '煞', '计', '剽', '臾', '抗', '防', '朦', '箔', '今', '扣', '柩', '绎', '长', '殒', '孕', '周', '株', '祠', '祷', '斡', '赠', '酣', '剃', '九', '色', '攘', '处', '喳', '胄', '翌', '穷', '时', '契', '窗', '唱', '蝴', '撒', '馓', '蝰', '识', '淀', '为', '诗', '忍', '廷', '茹', '亳', '涔', '旭', '拜', '死', '妯', '剩', '爱', '瀣', '墩', '腾', '伟', '埃', '接', '颌', '蟋', '抵', '坍', '无', '欲', '糌', '拳', '降', '功', '鳄', '诚', '生', '颅', '撤', '鞘', '喔', '蚊', '咆', '僵', '丙', '腿', '窍', '皓', '叛', '液', '坯', '彻', '怎', '熏', '悬', '婆', '侮', '度', '鹦', '优', '讨', '游', '咬', '绯', '瑀', '逊', '砍', '跟', '意', '掣', '卖', '饨', '喊', '炫', '职', '汇', '锰', '黝', '黯', '烈', '苑', '瞎', '耄', '蜡', '磅', '硅', '崇', '乃', '煤', '麒', '囔', '村', '囊', '球', '韩', '哩', '厅', '载', '愁', '志', '交', '角', '嗔', '贤', '璃', '筝', '滩', '榫', '僧', '狈', '眼', '碧', '魁', '熄', '脸', '能', '吆', '礁', '苗', '余', '煦', '哥', '嗟', '乍', '馨', '跋', '仆', '佛', '呼', '聂', '登', '甥', '领', '呕', '党', '频', '莺', '猫', '管', '泡', '需', '距', '猩', '椒', '嗓', '沌', '恳', '乏', '睍', '舌', '圳', '凡', '坚', '祝', '戈', '冢', '瑞', '掺', '稀', '蚁', '匠', '叵', '农', '狸', '臼', '享', '社', '嫂', '嫣', '寓', '谱', '湳', '佼', '唇', '咙', '阴', '涅', '莆', '悚', '订', '怂', '应', '岑', '惩', '恁', '咎', '隙', '偲', '揆', '嘟', '个', '命', '拈', '刮', '迅', '亘', '聆', '凰', '委', '如', '栖', '杀', '砚', '歪', '荷', '框', '烧', '幕', '鹏', '窑', '廉', '挑', '物', '眉', '赶', '鸽', '匡', '迷', '转', '荪', '孔', '陇', '粟', '穿', '裹', '里', '淇', '谬', '汉', '债', '禺', '瞿', '却', '密', '巨', '毛', '贷', '诞', '晓', '困', '置', '蚬', '恨', '划', '堪', '搬', '蔓', '彪', '枱', '峁', '丞', '至', '佟', '条', '坎', '锢', '咗', '霸', '妒', '敬', '迂', '呗', '畔', '痛', '签', '坷', '玫', '全', '廖', '咩', '浇', '语', '垄', '涧', '卫', '肭', '佣', '嗜', '场', '鞑', '过', '飓', '稿', '脊', '得', '奕', '串', '舷', '科', '韶', '麻', '摔', '吞', '骗', '兽', '矫', '胖', '尔', '勉', '性', '秀', '尚', '寺', '峥', '肤', '撺', '孬', '满', '阻', '名', '辆', '吟', '蝠', '松', '扼', '室', '彤', '更', '烦', '隧', '消', '肉', '民', '起', '曲', '梢', '糍', '爹', '饥', '娆', '患', '吻', '囡', '遣', '促', '哮', '慌', '冤', '勿', '弈', '盛', '墅', '岔', '欧', '缈', '秦', '钓', '蓬', '拴', '嘹', '跬', '斜', '蓓', '椎', '嫄', '莎', '片', '耋', '蔑', '镯', '察', '煜', '据', '弘', '半', '欣', '芜', '壮', '匆', '娼', '惑', '捣', '吃', '瓜', '劈', '栾', '昝', '春', '抨', '衷', '脐', '尧', '奸', '蝶', '茆', '锥', '哪', '聚', '泗', '帛', '么', '迁', '赏', '差', '雄', '戴', '蜱', '希', '迥', '部', '郊', '售', '嗣', '徒', '骄', '夸', '办', '桨', '嵬', '卑', '邹', '偷', '帐', '犇', '诋', '仁', '缘', '茵', '植', '痞', '涪', '掇', '首', '后', '循', '喜', '嘿', '运', '戟', '郴', '鸿', '桓', '缴', '指', '寨', '莞', '兜', '肋', '合', '孵', '州', '绺', '扩', '填', '努', '章', '兆', '卦', '瞩', '质', '妥', '羹', '炊', '浙', '落', '购', '见', '琦', '寿', '楞', '稼', '伽', '襟', '米', '曰', '颧', '朗', '妨', '亥', '祎', '缉', '癔', '嘢', '艾', '撇', '工', '孰', '畜', '寸', '驻', '须', '沈', '杞', '价', '坡', '缝', '请', '垢', '琵', '窕', '蓊', '仅', '桶', '悉', '侄', '架', '虽', '父', '暾', '涡', '虫', '掌', '檩', '埋', '黢', '苞', '聋', '稻', '扶', '佳', '嘭', '鑫', '廪', '饱', '雌', '维', '俭', '妓', '队', '残', '鸦', '鲨', '荆', '瑕', '锏', '衬', '芮', '闪', '蔫', '持', '展', '侥', '怠', '婥', '钺', '幂', '趸', '绒', '刑', '肘', '宅', '臊', '量', '偶', '胳', '梓', '吖', '娓', '蒂', '挥', '魄', '枯', '气', '财', '恰', '赛', '屿', '铤', '璀', '业', '拨', '驱', '狱', '顾', '训', '砾', '团', '很', '最', '百', '污', '超', '刹', '袜', '省', '尬', '肢', '赣', '捋', '释', '掠', '仑', '蒜', '眷', '沉', '溪', '蟆', '喲', '忖', '妙', '梧', '腊', '每', '挡', '赚', '眺', '倚', '灌', '柿', '椁', '馅', '自', '熀', '郭', '录', '辩', '靼', '瘦', '圈', '磋', '竞', '跌', '惴', '申', '牢', '邪', '黎', '争', '笔', '惜', '铝', '骐', '遗', '话', '紊', '鼠', '灾', '呱', '遭', '蹑', '编', '霜', '瘸', '赫', '掂', '屏', '杭', '扑', '赃', '陶', '约', '诉', '饿', '份', '帚', '塔', '蠢', '麟', '冷', '焱', '拒', '搂', '彰', '病', '险', '腐', '杈', '噼', '碎', '钗', '嗦', '歧', '啬', '饭', '司', '霎', '踌', '馋', '娩', '尹', '勇', '中', '亡', '企', '宾', '捐', '课', '蟹', '冶', '渠', '烛', '擎', '广', '担', '擞', '痣', '庸', '郦', '亦', '囵', '拽', '甩', '笼', '定', '旅', '捉', '逝', '荣', '睹', '掘', '蜻', '栻', '说', '治', '哽', '厝', '袄', '捕', '勾', '窄', '肃', '批', '店', '诱', '匀', '餐', '蚱', '憨', '旱', '忠', '贬', '簇', '婍', '凤', '王', '桔', '捷', '噢', '奖', '澈', '倍', '诽', '囚', '检', '伦', '玘', '参', '地', '睾', '鹂', '杨', '屯', '只', '厉', '独', '振', '花', '溃', '咪', '抑', '瓮', '走', '捞', '陡', '龚', '疙', '肌', '颖', '城', '晟', '遂', '蕙', '肴', '恶', '摘', '公', '芭', '澄', '哆', '唁', '肥', '免', '腴', '撼', '酋', '蹿', '掀', '主', '之', '受', '调', '幅', '阔', '柴', '瘤', '挞', '基', '泰', '钩', '岭', '驰', '冥', '涌', '巅', '獒', '棒', '笤', '湘', '帅', '品', '穗', '沆', '绅', '胆', '称', '鬓', '放', '念', '饕', '喷', '傅', '迎', '这', '猛', '婚', '津', '乒', '甘', '码', '槐', '哼', '敖', '给', '分', '隘', '擅', '拢', '小', '专', '暗', '莽', '鳅', '虾', '呐', '札', '鲤', '依', '估', '私', '辑', '续', '屋', '讲', '兄', '孤', '浴', '峡', '特', '浓', '套', '赵', '行', '沿', '鲫', '衡', '哝', '了', '嘲', '原', '暧', '产', '兑', '什', '夫', '琼', '秧', '纤', '嫁', '迸', '鹭', '叩', '拆', '被', '虞', '嫩', '雯', '充', '抖', '台', '闷', '跛', '鳗', '粤', '门', '惯', '趟', '袅', '箭', '骤', '许', '蟒', '繁', '房', '鳖', '雪', '窈', '锁', '近', '猜', '美', '辟', '喘', '纺', '补', '休', '侣', '辙', '巾', '储', '益', '埠', '骑', '火', '恺', '淼', '诩', '男', '励', '茜', '钰', '始', '助', '渭', '料', '疏', '蓄', '雀', '译', '蛙', '照', '烟', '境', '铜', '盏', '旬', '沂', '赧', '区', '槽', '谀', '或', '潇', '抽', '叱', '施', '联', '劣', '肯', '碟', '妖', '黍', '狒', '淙', '镜', '皇', '坛', '他', '席', '捡', '傣', '紫', '师', '炖', '在', '弥', '迦', '譬', '笛', '陈', '嘎', '医', '唵', '玻', '舅', '仨', '捶', '臣', '叹', '额', '低', '捻', '恐', '喽', '痍', '森', '蕤', '炸', '揿', '立', '梨', '巢', '哗', '仕', '弊', '蜢', '堡', '荐', '驶', '劭', '锹', '成', '况', '鳔', '豺', '剁', '拾', '龇', '女', '乐', '由', '建', '桥', '界', '描', '集', '进', '束', '杂', '灰', '盔', '然', '煮', '蹬', '饲', '倘', '钙', '教', '剧', '嘴', '旸', '儒', '何', '厦', '蚂', '租', '诺', '膝', '懈', '酱', '榄', '泓', '因', '帘', '箱', '屹', '蜥', '吼', '泌', '谕', '捅', '汲', '蜷', '旋', '裴', '蔼', '唤', '伪', '蹉', '府', '赂', '汾', '詹', '梯', '愚', '躁', '电', '缅', '鸥', '咳', '噬', '弯', '骨', '腆', '茅', '期', '泼', '丧', '味', '雕', '敞', '闻', '歉', '矢', '酬', '锈', '镇', '完', '踩', '出', '幼', '院', '四', '衩', '杖', '喀', '洛', '晰', '哒', '跺', '罚', '竭', '具', '蜓', '镊', '诅', '釉', '淑', '资', '翼', '摊', '即', '哈', '轿', '浦', '婴', '筹', '注', '副', '艇', '潭', '讯', '讹', '睬', '咻', '祁', '碌', '滔', '战', '贪', '阀', '枸', '霆', '渥', '柚', '嚼', '贫', '冽', '拓', '先', '咖', '裱', '啡', '醇', '尝', '固', '叶', '赌', '艘', '衫', '崴', '毙', '肠', '骙', '嘱', '菲', '漏', '离', '详', '挨', '巳', '峰', '憾', '喧', '纂', '垛', '鳞', '魏', '幻', '笙', '及', '漠', '缜', '痤', '鼓', '瀚', '嬉', '醉', '赎', '征', '迭', '闫', '耿', '赁', '系', '败', '诈', '渤', '躲', '咛', '欺', '蹈', '茫', '法', '剂', '显', '咸', '捂', '渔', '导', '著', '瞥', '堰', '芽', '唏', '抬', '闭', '鸡', '凶', '怨', '岸', '蜴', '阶', '池', '嫉', '笨', '者', '蚀', '阿', '炘', '遛', '犒', '呻', '轮', '告', '询', '初', '彭', '培', '贡', '裁', '皎', '往', '唢', '睦', '少', '泪', '矩', '颁', '躇', '觉', '糯', '瞌', '咂', '均', '贴', '限', '逑', '阮', '娇', '琢', '鲸', '扒', '溥', '宏', '聪', '酝', '煃', '嵌', '那', '罢', '紧', '承', '破', '菜', '借', '卡', '崭', '嘚', '言', '呀', '舒', '呜', '囝', '浩', '傻', '措', '姓', '终', '仇', '汰', '媒', '杯', '搐', '勃', '来', '术', '谍', '泖', '脉', '敕', '冀', '婀', '绪', '母', '替', '龙', '氛', '貌', '蕊', '宜', '潦', '道', '猎', '荫', '弩', '投', '吩', '盘', '肱', '腔', '程', '寀', '谆', '浑', '德', '筒', '级', '蛛', '磕', '踹', '存', '叼', '旺', '齿', '筐', '訾', '毁', '歹', '述', '擦', '侯', '瘪', '偏', '画', '潮', '璟', '鳝', '鸪', '乔', '式', '仞', '琏', '芙', '佯', '淘', '迹', '蛋', '裸', '踵', '苒', '楼', '星', '漾', '华', '祯', '袍', '蘑', '瓤', '刻', '缰', '矿', '帆', '关', '鼾', '溁', '奥', '晴', '崚', '堑', '求', '执', '拍', '鞋', '墓', '辞', '徙', '裕', '兮', '蚶', '轩', '园', '曹', '娘', '竖', '摧', '稣', '嵯', '六', '忾', '腼', '乌', '惊', '献', '缭', '甫', '捧', '誉', '铰', '辖', '攸', '噶', '损', '珊', '棉', '俺', '浏', '帝', '护', '形', '莲', '踏', '源', '费', '痊', '妃', '馕', '珠', '看', '劫', '悟', '铿', '簪', '到', '叭', '忙', '外', '儿', '蚣', '瑾', '粹', '贾', '砰', '骋', '屌', '淋', '跃', '泯', '亓', '遮', '鹊', '抛', '卿', '煲', '例', '断', '闽', '漩', '龟', '思', '镀', '卸', '耸', '档', '收', '邮', '雏', '朔', '负', '景', '织', '牖', '环', '喉', '氓', '淹', '荧', '毫', '慨', '验', '彼', '邃', '刊', '撩', '又', '朋', '戛', '谊', '引', '汗', '乎', '涵', '宫', '涂', '统', '北', '匝', '筋', '桂', '橱', '孱', '势', '茄', '博', '辰', '臭', '噪', '竿', '墙', '窦', '刀', '括', '涕', '赞', '仍', '咏', '摄', '临', '伊', '厚', '堵', '螃', '陌', '扁', '槛', '哺', '藐', '芬', '虔', '伸', '饮', '栉', '棵', '惟', '悯', '面', '爆', '崔', '磊', '疹', '西', '枉', '班', '垦', '匹', '槟', '皂', '间', '萝', '霄', '喱', '蜀', '提', '暄', '第', '撕', '惦', '喻', '猿', '智', '孩', '耕', '纠', '煎', '减', '帧', '鲶', '酊', '兵', '默', '吹', '抓', '盎', '序', '徐', '刨', '蹋', '绷', '住', '浅', '哋', '疚', '跪', '刷', '旯', '瑟', '咯', '糗', '刚', '世', '素', '渝', '考', '鹉', '以', '噜', '葬', '颈', '浆', '踊', '蝉', '谤', '抡', '赡', '空', '除', '夯', '愉', '嚷', '陀', '疼', '酉', '泱', '键', '耷', '聘', '篮', '俨', '愈', '逡', '庭', '砣', '杠', '筑', '谎', '偿', '貂', '腕', '峨', '侠', '琐', '责', '甜', '服', '七', '也', '壕', '豁', '滇', '铃', '惫', '勤', '甭', '腺', '胚', '奶', '骡', '衅', '魔', '夏', '濠', '岛', '炬', '控', '窃', '纸', '同', '幌', '嘣', '朽', '木', '酸', '淄', '当', '标', '糙', '翟', '壁', '历', '排', '燮', '庐', '肽', '妇', '薪', '堂', '久', '熙', '睛', '萧', '盟', '灿', '尴', '舶', '枇', '商', '缤', '碜', '痘', '狭', '丛', '贵', '惰', '扳', '驯', '河', '夷', '似', '谐', '匿', '坟', '掐', '叉', '蹦', '琛', '葱', '座', '剿', '吏', '把', '赋', '共', '讷', '砌', '射', '芦', '倡', '获', '庙', '厄', '育', '净', '宝', '咒', '函', '营', '港', '菊', '粱', '垂', '障', '疾', '恭', '段', '茂', '炼', '会', '遢', '抱', '庞', '摇', '蛤', '罐', '垃', '恙', '滋', '搁', '颓', '逃', '穆', '袱', '扭', '退', '肇', '予', '晾', '斥', '粑', '泄', '飞', '冠', '燃', '圾', '宗', '香', '莫', '蛟', '挛', '彬', '翔', '混', '搡', '衔', '戗', '喂', '赴', '嘞', '顶', '榻', '舍', '还', '宴', '她', '瓦', '淌', '感', '缪', '礼', '尻', '趔', '褐', '嚣', '哦', '滞', '田', '泻', '买', '候', '待', '蜈', '劝', '耘', '累', '覆', '絮', '栽', '澎', '佬', '逍', '愣', '拐', '授', '逗', '肾', '蟮', '目', '变', '滂', '泥', '决', '懦', '犯', '胥', '鹕', '亵', '口', '暖', '增', '唧', '慢', '溺', '占', '锵', '乖', '御', '映', '沃', '坦', '霏', '兹', '赟', '塞', '刃', '忐', '督', '泞', '刁', '坐', '啤', '祀', '壅', '哑', '巧', '岖', '窥', '枪', '汹', '胸', '国', '滚', '吶', '辕', '榜', '蔽', '罗', '链', '樟', '网', '耍', '舀', '伞', '谴', '髦', '致', '新', '姆', '镉', '褡', '蜂', '骂', '刳', '鹤', '纳', '艺', '写', '巩', '利', '尾', '题', '试', '妆', '石', '卵', '咔', '嗽', '醋', '瓷', '奠', '缮', '坪', '搏', '蔡', '嘶', '拘', '靠', '欠', '荒', '漓', '渲', '带', '翁', '晖', '綪', '干', '玩', '擘', '缔', '沧', '疃', '葩', '窨', '红', '芹', '靖', '昙', '床', '忌', '坠', '杷', '彖', '届', '示', '沱', '吱', '羔', '捎', '杳', '胀', '驳', '绵', '鹿', '嬢', '书', '濮', '赖', '旨', '纹', '蜿', '劲', '继', '霉', '谦', '铟', '淫', '熬', '婕', '恿', '逐', '酌', '厢', '蜕', '哇', '配', '乱', '愕', '珀', '狡', '威', '容', '莱', '圆', '某', '屑', '造', '啃', '祸', '洋', '鸩', '议', '裤', '晋', '械', '陆', '站', '錾', '韧', '隆', '月', '纯', '姿', '毯', '毕', '永', '幛', '入', '穰', '疲', '相', '象', '蔚', '回', '钳', '返', '锋', '英', '鸟', '踢', '咤', '几', '姻', '薇', '尖', '态', '缓', '柬', '腽', '介', '巍', '恒', '毂', '晕', '溜', '它', '娱', '馍', '渡', '此', '堕', '塄', '棺', '簧', '誓', '理', '餮', '彷', '兴', '砂', '勘', '裂', '瑶', '烫', '翩', '盲', '缙', '戒', '缵', '姬', '揉', '勋', '洱', '昧', '瑚', '咧', '涨', '猪', '税', '仙', '蝇', '斗', '烙', '邯', '颇', '璋', '莜', '垚', '饰', '橙', '绣', '毋', '埂', '次', '袋', '栓', '移', '隔', '梅', '送', '吊', '搪', '曳', '湖', '伴', '尸', '奭', '耳', '鉴', '饪', '镂', '户', '青', '挠', '陨', '矮', '娅', '络', '茨', '柞', '没', '沓', '诫', '递', '祥', '雇', '胱', '羸', '黛', '朝', '妤', '平', '蛳', '菀', '率', '柏', '氨', '锡', '绞', '夭', '濉', '茗', '稍', '纱', '馁', '羌', '蜗', '媚', '易', '盼', '慎', '诶', '笆', '吒', '秽', '丰', '幸', '锄', '赢', '沣', '骆', '宣', '镖', '乙', '暑', '军', '撅', '影', '糊', '勒', '附', '耽', '祖', '癸', '灶', '重', '汁', '俵', '元', '彩', '樱', '氢', '耀', '炭', '臀', '帽', '微', '儡', '椅', '绸', '渣', '鼎', '字', '粥', '吾', '底', '追', '沦', '篓', '蚌', '沟', '羚', '孑', '阁', '胡', '废', '删', '绕', '板', '荏', '驭', '连', '霍', '策', '琪', '烨', '栏', '沫', '浸', '崎', '跨', '症', '怒', '含', '该', '汝', '鹅', '侨', '胧', '乳', '剔', '直', '蹭', '萨', '狮', '伶', '碾', '舵', '涎', '洗', '仓', '榣', '翠', '姐', '碍', '吝', '诀', '炜', '酿', '属', '适', '剑', '悍', '陷', '剐', '加', '将', '徊', '究', '署', '薮', '寝', '箕', '狠', '蔗', '止', '怖', '躺', '垠', '殃', '宋', '设', '传', '壹', '井', '崩', '仿', '晤', '扪', '寅', '揽', '揪', '芳', '亢', '谈', '太', '尤', '俄', '绚', '疗', '墟', '路', '现', '袖', '玛', '颐', '轧', '丁', '犹', '殡', '辛', '茁', '蕉', '僚', '谄', '代', '谅', '蒋', '毒', '福', '蓉', '淡', '孙', '榨', '仲', '川', '倾', '曝', '饽', '实', '饶', '背', '反', '未', '狗', '型', '三', '贞', '判', '啫', '谋', '削', '懂', '窿', '辽', '葛', '食', '状', '核', '芝', '嫦', '膳', '票', '煌', '臂', '呵', '比', '些', '痉', '梦', '渍', '蔷', '光', '铁', '折', '贸', '可', '绩', '瑰', '末', '咱', '瓿', '庆', '惬', '蕴', '歙', '暨', '艰', '隐', '磁', '泉', '竟', '诿', '哟', '横', '锅', '披', '琉', '粘', '召', '唆', '膏', '峒', '逢', '惨', '化', '镕', '跄', '词', '操', '万', '簿', '趴', '梵', '谷', '贝', '灯', '羽', '老', '针', '囤', '归', '龄', '打', '圩', '拥', '难', '堤', '悦', '匪', '吗', '囫', '靡', '懒', '疆', '闸', '数', '布', '斧', '精', '啦', '搀', '薛', '触', '尿', '冒', '匾', '幢', '凝', '冈', '币', '波', '冬', '保', '岂', '遥', '磺', '析', '风', '斌', '作', '练', '端', '猥', '赔', '做', '爷', '棠', '轶', '凄', '謇', '昵', '齐', '跎', '妄', '块', '硫', '拿', '唯', '丢', '兼', '切', '眯', '靶', '韬', '县', '虚', '压', '证', '隋', '焊', '粪', '怦', '琰', '碑', '秒', '滑', '拯', '款', '婶', '亭', '讥', '撮', '惭', '缺', '胶', '仔', '爪', '朱', '托', '膛', '市', '发', '根', '媳', '巷', '讽', '痒', '皮', '急', '帜', '洼', '掳', '叙', '嘘', '砖', '盹', '汤', '攥', '嫚', '摹', '覃', '常', '浮', '寐', '茶', '黔', '屠', '桐', '卜', '溶', '逼', '斐', '丑', '颜', '肝', '洙', '逸', '馆', '寞', '枢', '厕', '吁', '贻', '骥', '经', '丽', '啪', '脖', '震', '旧', '帷', '贿', '侦', '廓', '藤', '璎', '阐', '股', '熟', '越', '蒸', '甚', '徨', '筇', '问', '舐', '豹', '与', '犄', '焕', '渗', '狍', '一', '派', '卢', '稚', '绑', '扎', '嬿', '凯', '俩', '亿', '哨', '快', '迪', '胁', '骼', '斩', '撸', '峦', '郑', '俱', '捱', '贱', '嘉', '哐', '千', '癫', '娥', '强', '握', '昨', '敷', '怜', '戚', '啥', '吽', '薏', '咋', '湃', '好', '淆', '围', '瘫', '浐', '驾', '悄', '盐', '皱', '撬', '喝', '殿', '歌', '腩', '溉', '艮', '寇', '棘', '染', '禄', '淮', '拌', '丹', '怀', '阅', '厨', '檐', '格', '热', '篡', '怕', '氏', '远', '嗨', '让', '随', '于', '头', '阜', '肚', '慕', '藏', '囱', '珍', '割', '冰', '靳', '任', '坝', '怄', '莉', '贼', '岗', '遵', '寄', '演', '样', '版', '吕', '尘', '奴', '铺', '籁', '对', '下', '娶', '痪', '们', '向', '吨', '值', '所', '宙', '屁', '掩', '着', '昊', '研', '忑', '殖', '拼', '害', '选', '砟', '俐', '铭', '莓', '瞒', '忽', '妞', '蝙', '响', '供', '讧', '杆', '冇', '懵', '缩', '岌', '轻', '玉', '不', '革', '拟', '甸', '机', '炷', '呆', '痹', '阚', '挚', '猝', '漳', '溯', '肿', '蓝', '惹', '墨', '沾', '钱', '妻', '搞', '颠', '稠', '懊', '坤', '菌', '茸', '髻', '取', '付', '本', '燕', '蝼', '技', '蜒', '呛', '挣', '瞄', '吴', '奋', '颗', '渴', '草', '查', '禹', '仰', '换', '粒', '啜', '洞', '堆', '壶', '安', '警', '黑', '豚', '图', '觅', '牛', '掰', '推', '慧', '预', '凹', '血', '俘', '粮', '俪', '动', '珞', '凋', '卧', '帮', '荻', '探', '楹', '嗲', '渊', '兰', '戏', '假', '鲜', '拉', '客', '糖', '锻', '呢', '夕', '和', '绍', '妪', '募', '鬣', '癖', '再', '乘', '疤', '绿', '夹', '酷', '牧', '罔', '埔', '咕', '马', '抹', '霖', '温', '筱', '罂', '踱', '坑', '蝗', '秩', '麝', '绽', '眨', '丸', '谁', '葡', '涯', '瞻', '遇', '闺', '昆', '赐', '蛇', '土', '哲', '拗', '讼', '邕', '棋', '嘛', '徽', '禽', '鄂', '碰', '蹼', '添', '闹', '睐', '略', '宛', '尼', '瘾', '郸', '菱', '油', '睽', '趾', '弟', '活', '停', '燥', '晶', '绘', '恋', '节', '嗡', '洽', '屉', '昏', '芯', '嗑', '协', '静', '要', '弃', '睡', '员', '搜', '夺', '碳', '升', '琶', '滤', '骚', '萍', '泊', '模', '开', '饵', '禁', '沮', '蜜', '溢', '幺', '慈', '黏', '桩', '苯', '烘', '嫌', '果', '垒', '诧', '砸', '诡', '歇', '贺', '皆', '学', '蚤', '错', '像', '政', '羞', '獐', '碗', '妮', '瞅', '稳', '伎', '哀', '舞', '童', '吧', '篷', '郡', '忒', '枕', '忏', '狂', '嘻', '瘀', '鞍', '二', '苟', '羱', '厂', '陪', '钠', '锐', '铆', '枫', '雍', '袁', '涩', '犬', '伙', '傍', '庄', '辈', '铲', '瓣', '鲍', '婷', '峻', '捍', '讪', '羲', '辨', '域', '掬', '搅', '吉', '贩', '输', '坏', '靴', '的', '陋', '亏', '玲', '嚎', '援', '汶', '符', '读', '翰', '江', '迤', '娌', '销', '肖', '艳', '嘅', '吮', '摁', '勐', '屡', '婿', '伍', '高', '善', '痰', '邀', '顿', '牒', '媲', '雁', '柑', '必', '两', '抠', '拧', '陵', '枝', '确', '疮', '八', '蝻', '昔', '违', '恬', '子', '滴', '榔', '泵', '湾', '绝', '报', '攀', '虐', '禅', '霭', '畅', '拎', '启', '搭', '奢', '项', '琳', '俑', '线', '丘', '酩', '啊', '宇', '冲', '跤', '东', '涉', '乞', '衣', '糟', '枣', '轨', '彝', '谣', '践', '槌', '翘', '狐', '而', '肩', '亩', '沛', '寂', '垴', '俗', '挺', '拂', '炒', '橘', '日', '云', '逵', '尕', '钟', '倔', '斋', '葵', '逻', '坳', '敢', '圃', '窜', '柜', '跑', '裙', '鸭', '句', '速', '蘸', '谓', '梳', '睿', '塱', '羡', '赘', '妍', '阳', '渺', '钦', '复', '禧', '页', '瞰', '绛', '够', '笑', '盗', '萦', '京', '揍', '刘', '散', '羊', '十', '伯', '舰', '蚕', '昂', '媃', '剖', '董', '厘', '搓', '押', '山', '豆', '孟', '偃', '莹', '耗', '韵', '杜', '扫', '大', '铅', '亮', '陉', '挟', '携', '殂', '构', '令', '卷', '炳', '飙', '乜', '声', '孝', '弋', '壳', '泣', '弱', '五', '寡', '敌', '龊', '殚', '挂', '涮', '晨', '帖']

symbols = [_pad, _eos] + list(_puncts) + _characters